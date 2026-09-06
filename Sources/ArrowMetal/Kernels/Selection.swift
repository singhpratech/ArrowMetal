import Foundation
import Metal

/// The remaining Arrow selection, sort and random functions, on the GPU.
///
/// **`inverse_permutation` and `scatter`.** One atomic scatter kernel builds the inverse of an index
/// column: for the `i`-th index the `index`-th output is `i`. Output length is `max_index + 1`, or the
/// input length when `max_index` is negative. A slot no index names comes back **null**, and when
/// several positions name the same slot the **last** one wins — Arrow's rule, and here it is also
/// deterministic, because "last wins" is an `atomic_fetch_max` over the source positions and a maximum
/// does not care what order the threads arrive in. Null indices are ignored; an index outside
/// `[0, max_index]` raises.
///
/// `scatter(values, indices)` is that inverse permutation used as a `take`, so one kernel plus the
/// existing gather covers every element type, nested ones included: `take` already turns a null index
/// into a null output row, which is exactly what an unassigned slot must produce.
///
/// **`winsorize`.** Two quantiles from one GPU sort, then a clamp kernel. Arrow's limits are the
/// *nearest* quantiles, never interpolated ones: with `m` non-null, non-NaN values sorted ascending,
/// a limit `q` picks `sorted[round(q * (m - 1))]` with the halfway case going to the even index, so
/// both bounds are values that occur in the data. Nulls stay null and NaNs pass through (they take
/// part in neither the limits nor the comparison).
///
/// **`rank_quantile` and `rank_normal`.** One argsort, run marks over the sorted order, an int32 scan
/// of those marks, the run bounds and a scatter back to the original rows — the same shape as the
/// ranking family in `Kernels/Window.swift`, but the value a row gets is
/// `(average 1-based rank of its tie group - 0.5) / n`, which for a run over sorted positions `[s, e)`
/// is `(s + e) / (2n)`. Nulls sort last and form one tie group, matching pyarrow's default
/// `null_placement = "at_end"`; NaN is one value ordered after +inf, also matching pyarrow.
/// `rank_normal` is the normal percent-point function of that quantile.
///
/// **`random`.** Philox4x32-10, a counter-based generator: value `i` depends only on the counter `i`
/// and the seed, never on how the work was scheduled.
extension MetalArray {

    // MARK: - Inverse permutation and scatter

    /// Arrow `inverse_permutation`: for the `i`-th value in this index column, the `index`-th output
    /// element is `i`.
    ///
    /// The output has `maxIndex + 1` elements, or this column's length when `maxIndex` is negative.
    /// Slots no index names are null; duplicate indices resolve to the **last** (largest) source
    /// position, as Arrow does; null indices are skipped. An index outside `[0, maxIndex]` throws.
    ///
    /// The result is always `int32` — Arrow's `output_type` option is not implemented, and this
    /// library caps arrays at 2^32 elements, so an int32 position is always enough.
    public func inversePermutation(maxIndex: Int64 = -1) throws -> MetalArray<Int32> {
        guard !T.isFloatingPoint else {
            throw ArrowMetalError.unsupportedType("inverse_permutation needs an integer index column, got \(T.arrowFormat)")
        }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let outLen = maxIndex < 0 ? n : Int(maxIndex) + 1
        guard outLen >= 0, outLen <= Int(UInt32.max) else {
            throw ArrowMetalError.invalidArrowArray("inverse_permutation max_index \(maxIndex) is out of range")
        }
        let limit = Int64(outLen) - 1
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(outLen, 1) * 4, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(outLen, 1), zeroed: false, context: ctx)
        let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        let src = SelectionSource.indexOps(V: T.mslType)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "selection", source: src, function: f, type: T.mslType)
        }
        let fillPSO = try pso("sel_fill_i32"), scatterPSO = try pso("sel_invperm"), finishPSO = try pso("sel_invperm_finish")
        let hasV = validity != nil
        try ctx.run { enc in
            if outLen > 0 {
                enc.setComputePipelineState(fillPSO)
                enc.setBuffer(out.mtl, offset: out.offset, index: 0)
                Dispatch.setLength(enc, outLen, nil, index: 1)
                Dispatch.setScalar(enc, Int32(-1), index: 2)
                Dispatch.dispatch1D(enc, fillPSO, count: outLen)
                enc.memoryBarrier(scope: .buffers)
            }
            if n > 0 {
                enc.setComputePipelineState(scatterPSO)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                let v = validity ?? values
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, hasV ? 1 : 0, index: 3)
                var lim = limit
                enc.setBytes(&lim, length: 8, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                enc.setBuffer(errorFlag.mtl, offset: errorFlag.offset, index: 6)
                Dispatch.dispatch1D(enc, scatterPSO, count: n)
                enc.memoryBarrier(scope: .buffers)
            }
            if outLen > 0 {
                enc.setComputePipelineState(finishPSO)
                enc.setBuffer(out.mtl, offset: out.offset, index: 0)
                Dispatch.setLength(enc, outLen, nil, index: 1)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 2)
                Dispatch.dispatch1D(enc, finishPSO, count: outLen)
            }
        }
        try ctx.afterFlush { [errorFlag] in
            if errorFlag.typed(UInt32.self)[0] != 0 {
                throw ArrowMetalError.invalidArrowArray("inverse_permutation: index out of range [0, \(limit)]")
            }
        }
        ctx.retainUntilFlush(self); ctx.retainUntilFlush(errorFlag); ctx.retainUntilFlush(validBytes)
        let bm = outLen > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: outLen) : nil
        let res = MetalArray<Int32>(length: outLen, nullCount: 0, validity: bm, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// Arrow `scatter`: places the `i`-th value at the position named by the `i`-th index.
    ///
    /// The output has `maxIndex + 1` elements, or this column's length when `maxIndex` is negative.
    /// A position no index names is null, and on duplicate indices the last value wins.
    public func scattered<I: ArrowPrimitive>(to indices: MetalArray<I>, maxIndex: Int64 = -1) throws -> MetalArray<T> {
        guard indices.length == length else { throw ArrowMetalError.lengthMismatch(length, indices.length) }
        return try take(try indices.inversePermutation(maxIndex: maxIndex))
    }

    // MARK: - Winsorize

    /// Arrow `winsorize`: values below the lower quantile take the lower quantile's value and values
    /// above the upper one take the upper quantile's value, which trims the influence of outliers
    /// without dropping rows.
    ///
    /// The limits are Arrow's *nearest* quantiles, not interpolated ones. With `m` the number of
    /// non-null, non-NaN values in ascending order, a limit `q` picks `sorted[round(q * (m - 1))]`,
    /// the halfway case rounding to the even index — the same rule as
    /// `pyarrow.compute.quantile(interpolation="nearest")` — so both bounds are values that actually
    /// occur. Nulls are ignored and stay null; NaN is ignored and passes through, since it compares
    /// false against both bounds. An all-null (or all-NaN) column comes back unchanged.
    public func winsorize(lowerLimit: Double, upperLimit: Double) throws -> MetalArray<T> {
        guard lowerLimit >= 0, lowerLimit <= 1, upperLimit >= 0, upperLimit <= 1, lowerLimit <= upperLimit else {
            throw ArrowMetalError.invalidArrowArray("winsorize limits must satisfy 0 <= lower <= upper <= 1, got \(lowerLimit), \(upperLimit)")
        }
        let ctx = context
        let n = length
        guard n > 0 else { return self }
        let sortedValues = try sorted()
        let m = try Self.nonNullNonNaNCount(sortedValues, validCount: n - nullCount)
        guard m > 0 else { return self }
        let lowIndex = Self.nearestQuantileIndex(lowerLimit, count: m)
        let highIndex = Swift.max(lowIndex, Self.nearestQuantileIndex(upperLimit, count: m))
        let p = sortedValues.valuePointer
        let lo = p[lowIndex], hi = p[highIndex]
        let kind: SelectionSource.Kind = T.self == Double.self ? .float64 : (T.isFloatingPoint ? .float32 : .integer)
        // Metal has no `double`: a float64 column travels through the clamp as raw binary64 patterns.
        let mslT = T.self == Double.self ? "ulong" : T.mslType
        let src = SelectionSource.clamp(V: mslT, kind: kind)
        let pso = try Dispatch.pipeline(ctx, family: "selection-clamp", source: src, function: "sel_clamp", type: mslT)
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            Dispatch.setScalar(enc, lo, index: 2)
            Dispatch.setScalar(enc, hi, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        ctx.retainUntilFlush(self)
        return MetalArray<T>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Position of Arrow's `nearest` quantile `q` among `count` sorted values: `round(q * (count - 1))`
    /// with the halfway case going to the even index, clamped into range.
    static func nearestQuantileIndex(_ q: Double, count m: Int) -> Int {
        Swift.min(m - 1, Swift.max(0, Int((q * Double(m - 1)).rounded(.toNearestOrEven))))
    }

    /// How many of the sorted values are neither null nor NaN. `argsort` puts the nulls after every
    /// value and NaN immediately before them, so the NaN run is a suffix of `[0, validCount)` and a
    /// binary search finds where it starts.
    private static func nonNullNonNaNCount(_ sortedValues: MetalArray<T>, validCount: Int) throws -> Int {
        guard T.isFloatingPoint, validCount > 0 else { return validCount }
        let p = sortedValues.valuePointer
        func isNaN(_ i: Int) -> Bool { p[i].asDouble.isNaN }
        guard isNaN(validCount - 1) else { return validCount }
        var lo = 0, hi = validCount - 1                    // hi is NaN, lo may or may not be
        if isNaN(0) { return 0 }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if isNaN(mid) { hi = mid } else { lo = mid }
        }
        return hi
    }

    // MARK: - Quantile and normal ranks

    /// Arrow `rank_quantile`: the quantile rank of each row, strictly between 0 and 1.
    ///
    /// A row's value is `(average 1-based rank of its tie group - 0.5) / n`, the definition pyarrow
    /// uses. Nulls sort last and form one tie group (pyarrow's default `null_placement = "at_end"`),
    /// NaN is one value ordered after every other value but before the nulls, and the result is never
    /// itself null. Computed as `(s + e) / (2n)` over the run's sorted positions `[s, e)` with the
    /// correctly rounded software binary64 divide, so it matches a host `Double` division bit for bit.
    ///
    /// Arrow's full option surface is here: `descending` picks the direction of the single sort key and
    /// `nullPlacement` decides which end the nulls sit at (they are one tie group either way).
    public func rankQuantile(descending: Bool = false,
                             nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Double> {
        let ctx = context
        guard let o = try rankOrder(descending: descending, nullPlacement: nullPlacement) else {
            return try MetalArray<Double>([Double](), context: ctx)
        }
        let out = try MetalArrowBuffer.allocate(byteCount: o.n * 8, zeroed: false, context: ctx)
        let p = try Dispatch.pipeline(ctx, family: "selection-rank", source: o.source,
                                      function: "sel_quantile_scatter", type: o.unsignedType)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(o.ord.values.mtl, offset: o.ord.values.offset, index: 0)
            enc.setBuffer(o.groups.values.mtl, offset: o.groups.values.offset, index: 1)
            enc.setBuffer(o.runStart.mtl, offset: o.runStart.offset, index: 2)
            enc.setBuffer(o.runEnd.mtl, offset: o.runEnd.offset, index: 3)
            Dispatch.setLength(enc, o.n, nil, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            Dispatch.dispatch1D(enc, p, count: o.n)
        }
        ctx.retainUntilFlush(o)
        return MetalArray<Double>(length: o.n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// Arrow `rank_normal` in float32: the normal percent-point function of `rankQuantile()`,
    /// evaluated on the GPU with Acklam's rational approximation plus one Halley refinement through
    /// `erfc`. Accurate to about a float32 ulp of the true quantile (roughly 1e-6 absolute over the
    /// range a rank can produce); the float64 form is `rankNormal()`.
    public func rankNormalFloat32(descending: Bool = false,
                                  nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Float> {
        let ctx = context
        guard let o = try rankOrder(descending: descending, nullPlacement: nullPlacement) else {
            return try MetalArray<Float>([Float](), context: ctx)
        }
        let out = try MetalArrowBuffer.allocate(byteCount: o.n * 4, zeroed: false, context: ctx)
        let p = try Dispatch.pipeline(ctx, family: "selection-rank", source: o.source,
                                      function: "sel_normal_scatter_f32", type: o.unsignedType)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(o.ord.values.mtl, offset: o.ord.values.offset, index: 0)
            enc.setBuffer(o.groups.values.mtl, offset: o.groups.values.offset, index: 1)
            enc.setBuffer(o.runStart.mtl, offset: o.runStart.offset, index: 2)
            enc.setBuffer(o.runEnd.mtl, offset: o.runEnd.offset, index: 3)
            Dispatch.setLength(enc, o.n, nil, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            Dispatch.dispatch1D(enc, p, count: o.n)
        }
        ctx.retainUntilFlush(o)
        return MetalArray<Float>(length: o.n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// Arrow `rank_normal` in float64: `rankQuantile()` on the GPU, then the normal percent-point
    /// function on the host through Wichura's AS 241 (`NormalQuantile.ppf`), which is accurate to
    /// about 1e-16 relative. Metal has no `double`, and the software binary64 in `DoubleMath` has no
    /// `log`/`exp`/`erfc`, so the inverse CDF itself is the one part of this that runs on the CPU.
    public func rankNormal(descending: Bool = false,
                           nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Double> {
        let q = try rankQuantile(descending: descending, nullPlacement: nullPlacement)
        let n = q.length
        let p = q.mutableValuePointer
        for i in 0..<n { p[i] = NormalQuantile.ppf(p[i]) }
        return q
    }

    /// One argsort plus the run bookkeeping every quantile-rank function shares.
    private func rankOrder(descending: Bool, nullPlacement: NullPlacement) throws -> RankOrder? {
        let ctx = context
        let n = length
        guard n > 0 else { return nil }
        try Dispatch.checkLength(n)
        // The null rows form one contiguous block of the sorted order at whichever end was asked for.
        let m = n - nullCount
        let nullLo = nullPlacement == .atEnd ? m : 0
        let nullHi = nullPlacement == .atEnd ? n : nullCount
        let uType = UniqueSource.unsignedType(width: T.byteWidth)
        let src = SelectionSource.ranking(U: uType)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "selection-rank", source: src, function: f, type: uType)
        }

        // Floats compare as raw bit patterns here, so normalise first: one NaN pattern, -0 becomes +0.
        var keyValues = values
        var keyArray: MetalArray<T> = self
        if T.isFloatingPoint {
            let norm = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
            let normPSO = try pso(T.byteWidth == 8 ? "sel_norm_f64" : "sel_norm_f32")
            try ctx.run { enc in
                enc.setComputePipelineState(normPSO)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(norm.mtl, offset: norm.offset, index: 2)
                Dispatch.dispatch1D(enc, normPSO, count: n)
            }
            ctx.retainUntilFlush(self)
            keyValues = norm
            keyArray = MetalArray<T>(length: n, nullCount: nullCount, validity: validity, values: norm, context: ctx)
        }

        let ord = try keyArray.argsort(descending: descending, nullPlacement: nullPlacement)
        let marks = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let marksPSO = try pso("sel_marks")
        try ctx.run { enc in
            enc.setComputePipelineState(marksPSO)
            enc.setBuffer(keyValues.mtl, offset: keyValues.offset, index: 0)
            enc.setBuffer(ord.values.mtl, offset: ord.values.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, nullLo, index: 3)
            enc.setBuffer(marks.mtl, offset: marks.offset, index: 4)
            Dispatch.setUInt(enc, nullHi, index: 5)
            Dispatch.dispatch1D(enc, marksPSO, count: n)
        }
        ctx.retainUntilFlush(keyValues); ctx.retainUntilFlush(ord)
        // Inclusive prefix sum of the marks numbers the tie groups from 1.
        let groups = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: marks, context: ctx)
            .cumulative(.sum)
        let runStart = try MetalArrowBuffer.allocate(byteCount: n * 4, context: ctx)
        let runEnd = try MetalArrowBuffer.allocate(byteCount: n * 4, context: ctx)
        let boundsPSO = try pso("sel_run_bounds")
        try ctx.run { enc in
            enc.setComputePipelineState(boundsPSO)
            enc.setBuffer(marks.mtl, offset: marks.offset, index: 0)
            enc.setBuffer(groups.values.mtl, offset: groups.values.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(runStart.mtl, offset: runStart.offset, index: 3)
            enc.setBuffer(runEnd.mtl, offset: runEnd.offset, index: 4)
            Dispatch.dispatch1D(enc, boundsPSO, count: n)
        }
        for o in [marks, runStart, runEnd] as [AnyObject] { ctx.retainUntilFlush(o) }
        ctx.retainUntilFlush(groups)
        return RankOrder(ord: ord, groups: groups, runStart: runStart, runEnd: runEnd, n: n,
                         source: src, unsignedType: uType)
    }

    // MARK: - first_last

    /// Arrow `first_last`: a one-row struct with fields `first` and `last`, each of this column's
    /// type. With `skipNulls` (the default) the first and last **non-null** values are used and the
    /// fields are null only when the column has no valid row; with `skipNulls: false` the first and
    /// last rows are taken as they are, null included.
    ///
    /// This is `first()` and `last()` — one GPU pass over the validity bitmap taking the atomic
    /// minimum and maximum valid index — packaged as the struct scalar Arrow returns.
    public func firstLast(skipNulls: Bool = true) throws -> MetalStructArray {
        let f = try first(skipNulls: skipNulls), l = try last(skipNulls: skipNulls)
        return try MetalStructArray(names: ["first", "last"],
                                    children: [AnyMetalArray.erasing(try MetalArray<T>([f], context: context)),
                                               AnyMetalArray.erasing(try MetalArray<T>([l], context: context))],
                                    context: context)
    }
}

/// The sorted order and per-run bookkeeping the quantile-rank functions share: one argsort, one scan.
final class RankOrder {
    /// Original row index of each sorted position (nulls last).
    let ord: MetalArray<Int32>
    /// 1-based tie-group number of each sorted position (the inclusive scan of the run marks).
    let groups: MetalArray<Int32>
    /// First and one-past-last sorted position of each tie group, indexed by its 0-based number.
    let runStart: MetalArrowBuffer
    let runEnd: MetalArrowBuffer
    let n: Int
    /// Generated MSL for this element width, and its pipeline cache key.
    let source: String
    let unsignedType: String

    init(ord: MetalArray<Int32>, groups: MetalArray<Int32>, runStart: MetalArrowBuffer,
         runEnd: MetalArrowBuffer, n: Int, source: String, unsignedType: String) {
        self.ord = ord; self.groups = groups; self.runStart = runStart; self.runEnd = runEnd
        self.n = n; self.source = source; self.unsignedType = unsignedType
    }
}

// MARK: - Type-erased selection

extension AnyMetalArray {
    /// Wraps a typed primitive array in the type-erased enum. The ten `ArrowPrimitive` conformers are
    /// exactly the ten primitive cases, so this is total.
    public static func erasing<T: ArrowPrimitive>(_ a: MetalArray<T>) -> AnyMetalArray {
        switch a {
        case let x as MetalArray<Int8>: return .int8(x)
        case let x as MetalArray<UInt8>: return .uint8(x)
        case let x as MetalArray<Int16>: return .int16(x)
        case let x as MetalArray<UInt16>: return .uint16(x)
        case let x as MetalArray<Int32>: return .int32(x)
        case let x as MetalArray<UInt32>: return .uint32(x)
        case let x as MetalArray<Int64>: return .int64(x)
        case let x as MetalArray<UInt64>: return .uint64(x)
        case let x as MetalArray<Float>: return .float32(x)
        case let x as MetalArray<Double>: return .float64(x)
        default: fatalError("AnyMetalArray.erasing: \(T.self) is not an Arrow primitive case")
        }
    }

    /// Arrow `scatter` for a column of any type: the `i`-th value goes to the position named by the
    /// `i`-th index. Unassigned positions are null, duplicate indices resolve to the last value.
    public func scattered<I: ArrowPrimitive>(to indices: MetalArray<I>, maxIndex: Int64 = -1) throws -> AnyMetalArray {
        guard indices.length == length else { throw ArrowMetalError.lengthMismatch(length, indices.length) }
        return try take(try indices.inversePermutation(maxIndex: maxIndex))
    }

    /// Arrow `count_all`: the number of rows, valid or not. O(1) metadata; inside an open batch,
    /// reading it forces a sync point exactly as `length` does.
    public var countAll: Int { length }

    /// Arrow `true_unless_null`: `true` for every valid row and `null` for every null one.
    ///
    /// The values bitmap is a `memset` on the host — every bit is 1 — and the validity bitmap is
    /// shared with the input with no copy, so no kernel runs. Union and run-end encoded columns
    /// throw: neither carries a top-level validity bitmap to share.
    public func trueUnlessNull() throws -> MetalBooleanArray {
        let n = length
        let ctx = context
        let bits = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), zeroed: false, context: ctx)
        memset(bits.mutableContents, 0xFF, Swift.max(Bitmap.byteCount(bits: n), 1))
        let v = try topLevelValidity()
        let res = MetalBooleanArray(length: n, nullCount: 0, validity: v, values: bits, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// The validity bitmap this column exposes at the top level, or nil when it has no nulls.
    private func topLevelValidity() throws -> MetalArrowBuffer? {
        switch self {
        case .int8(let a): return a.validity
        case .uint8(let a): return a.validity
        case .int16(let a): return a.validity
        case .uint16(let a): return a.validity
        case .int32(let a): return a.validity
        case .uint32(let a): return a.validity
        case .int64(let a): return a.validity
        case .uint64(let a): return a.validity
        case .float32(let a): return a.validity
        case .float64(let a): return a.validity
        case .boolean(let a): return a.validity
        case .string(let a): return a.validity
        case .temporal(let a): return a.validity
        case .binary(let a): return a.validity
        case .decimal(let a): return a.validity
        case .dictionary(let codes, _): return codes.validity
        case .list(let a): return a.validity
        case .structure(let a): return a.validity
        case .map(let a): return a.entries.validity
        case .union:
            throw ArrowMetalError.unsupportedType("true_unless_null: a union column has no top-level validity bitmap")
        case .runEndEncoded:
            throw ArrowMetalError.unsupportedType("true_unless_null: decode the run-end encoded column first")
        case .null: return nil
        case .float16(let a): return a.validity
        case .smallDecimal(let a): return a.validity
        case .interval(let a): return a.validity
        case .fixedBinary(let a): return a.validity
        case .extended(let e): return try e.storage.topLevelValidity()
        }
    }
}

// MARK: - Random

/// Arrow `random`: uniform float64 values in [0, 1) from a counter-based generator.
public enum ArrowRandom {
    /// `count` uniform float64 values in [0, 1), generated on the GPU with **Philox4x32-10**
    /// (Salmon, Moraes, Dror & Shaw, SC'11) keyed by `seed`.
    ///
    /// Element `i` is derived from the counter `(i, 0, 0, 0)` and the key `(seed low 32, seed high
    /// 32)`, so the sequence depends only on the seed — not on the threadgroup size, the device, or
    /// how the work was scheduled — and the same seed always gives the same array. Words 0 and 1 of
    /// the 128-bit Philox output form a 64-bit integer whose top 53 bits become a multiple of 2^-53
    /// in [0, 1), so every value is exactly representable and no value is ever 1.0.
    ///
    /// The stream is ArrowMetal's own; it does **not** reproduce the numbers Arrow C++ generates for
    /// the same seed (Arrow uses `pcg32_fast` on the host). Arrow's `"system"` initializer maps here
    /// to a seed drawn from the system random source.
    public static func uniform(count: Int, seed: UInt64, context: MetalContext = .shared) throws -> MetalArray<Double> {
        guard count >= 0 else { throw ArrowMetalError.invalidArrowArray("random count must be >= 0, got \(count)") }
        try Dispatch.checkLength(count)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(count, 1) * 8, zeroed: false, context: context)
        guard count > 0 else { return MetalArray<Double>(length: 0, nullCount: 0, validity: nil, values: out, context: context) }
        let pso = try Dispatch.pipeline(context, family: "selection-random", source: SelectionSource.random,
                                        function: "sel_random", type: "f64")
        try context.run { enc in
            enc.setComputePipelineState(pso)
            Dispatch.setLength(enc, count, nil, index: 0)
            var key = (UInt32(truncatingIfNeeded: seed), UInt32(truncatingIfNeeded: seed >> 32))
            enc.setBytes(&key, length: 8, index: 1)
            enc.setBuffer(out.mtl, offset: out.offset, index: 2)
            Dispatch.dispatch1D(enc, pso, count: count)
        }
        return MetalArray<Double>(length: count, nullCount: 0, validity: nil, values: out, context: context)
    }
}

// MARK: - Normal percent-point function

/// The inverse of the standard normal CDF on the host.
public enum NormalQuantile {
    /// Wichura's AS 241 `PPND16`: the normal percent-point function, accurate to about 1e-16 relative
    /// over the whole open interval. `ppf(0)` is -infinity, `ppf(1)` is +infinity and anything outside
    /// `[0, 1]` (NaN included) comes back NaN.
    public static func ppf(_ p: Double) -> Double {
        if p.isNaN || p < 0 || p > 1 { return Double.nan }
        if p == 0 { return -Double.infinity }
        if p == 1 { return Double.infinity }
        let q = p - 0.5
        if Swift.abs(q) <= 0.425 {
            let r = 0.180625 - q * q
            return q * poly(r, a) / poly(r, b)
        }
        var r = q < 0 ? p : 1 - p
        r = (-Foundation.log(r)).squareRoot()
        let value: Double
        if r <= 5 {
            let s = r - 1.6
            value = poly(s, c) / poly(s, d)
        } else {
            let s = r - 5
            value = poly(s, e) / poly(s, f)
        }
        return q < 0 ? -value : value
    }

    /// Horner over `coeffs[j] * x^j`.
    private static func poly(_ x: Double, _ coeffs: [Double]) -> Double {
        var acc = coeffs[coeffs.count - 1]
        for i in stride(from: coeffs.count - 2, through: 0, by: -1) { acc = acc * x + coeffs[i] }
        return acc
    }

    private static let a: [Double] = [3.3871328727963666080, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
                                      1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
                                      3.3430575583588128105e+4, 2.5090809287301226727e+3]
    private static let b: [Double] = [1.0, 4.2313330701600911252e+1, 6.8718700749205790830e+2,
                                      5.3941960214247511077e+3, 2.1213794301586595867e+4,
                                      3.9307895800092710610e+4, 2.8729085735721942674e+4,
                                      5.2264952788528545610e+3]
    private static let c: [Double] = [1.42343711074968357734, 4.63033784615654529590, 5.76949722146069140550,
                                      3.64784832476320460504, 1.27045825245236838258, 2.41780725177450611770e-1,
                                      2.27238449892691845833e-2, 7.74545014278341407640e-4]
    private static let d: [Double] = [1.0, 2.05319162663775882187, 1.67638483018380384940,
                                      6.89767334985100004550e-1, 1.48103976427480074590e-1,
                                      1.51986665636164571966e-2, 5.47593808499534494600e-4,
                                      1.05075007164441684324e-9]
    private static let e: [Double] = [6.65790464350110377720, 5.46378491116411436990, 1.78482653991729133580,
                                      2.96560571828504891230e-1, 2.65321895265761230930e-2,
                                      1.24266094738807843860e-3, 2.71155556874348757815e-5,
                                      2.01033439929228813265e-7]
    private static let f: [Double] = [1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1,
                                      1.48753612908506148525e-2, 7.86869131145613259100e-4,
                                      1.84631831751005468180e-5, 1.42151175831644588870e-7,
                                      2.04426310338993978564e-15]
}
