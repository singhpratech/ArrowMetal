import Foundation
import Metal

// The rest of Arrow's aggregate family: the grouped forms `Aggregates.swift` left out, and the three
// scalar ones it declared out of scope.
//
// | function | how it runs |
// |---|---|
// | `hash_min_max` | GPU, sort-free: two passes of 32-bit atomics over an order-preserving key |
// | `hash_count_all` | GPU, the existing group-by row count |
// | `hash_first_last` | GPU, two group-by extremes over a masked row index plus two gathers |
// | `hash_one` | GPU, a group-by minimum over the row index plus one gather |
// | `hash_list` | GPU, segmented gather after the stable sort by group id |
// | `hash_distinct` | GPU, dictionary encode + packed `unique` + segmented gather |
// | `hash_approximate_median` / `hash_quantile` | GPU sort by (group, value) plus a GPU per-group pick |
// | `hash_skew` / `hash_kurtosis` | two GPU passes in binary64: per-group means, then the 3rd/4th moments |
// | `hash_product` | GPU, segmented multiply reduction (was a host pass) |
// | `hash_pivot_wider` | GPU, one masked `hash_one` per pivot key |
// | `hash_tdigest` | GPU sort by (group, value), CPU merge of the centroids per group |
// | `skew` / `kurtosis` | two GPU passes, the same shape as the scalar `variance` |
// | `tdigest` | GPU sort; the digest of a sorted column is the column, so the quantile is read out of it |
//
// Precision. The fused `hash_min_max` is exact for every type. `hash_product` wraps in 64 bits for
// integers exactly as the scalar `product` does, multiplies Float32 in `float` and Float64 through the
// software binary64 routine, and reassociates across the threads of a group either way. `hash_skew` and
// `hash_kurtosis` now form their deviations about an **exact** per-group mean and accumulate them in
// software binary64 (`Kernels/GroupMoments.swift`), so they land near 1e-15 rather than the 1e-5 the
// Float32 deviations used to cost; the scalar
// `skew` and `kurtosis` use the same per-type deviation machinery as `variance` (64-bit integer part for
// integer columns, a two-float mean for Float32, software binary64 for Float64) and land near 1e-7 and
// 1e-15 respectively. `hash_approximate_median` is **exact**, not a sketch, for the same reason the
// scalar one is: sorting on the GPU is cheaper than approximating.

// MARK: - group-by: fused min/max, product, order statistics, moments

extension GroupBy {

    /// Arrow `hash_count_all`: rows per key, null values included. A key with no row is zero, never null.
    public func countAll() throws -> MetalArray<Int64> { try count() }

    /// Arrow `hash_count`: non-null values per key, for **any** value type including Float64.
    ///
    /// `count(_:)` reads the values through a typed kernel and so rejects Float64; counting only ever
    /// looks at the validity bitmap, so this reinterprets the values buffer as bytes and counts that.
    public func countValid<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let proxy = MetalArray<UInt8>(length: values.length, nullCount: values.nullCount,
                                      validity: values.validity, values: values.values, context: values.context)
        return try count(proxy)
    }

    /// Arrow `hash_min_max`: the smallest and largest non-null value of each key, from **one** read of
    /// the values. NaN is skipped, so a key whose only values are NaN is null, as in Arrow.
    ///
    /// GPU, sort-free: two linear passes of 32-bit atomics over an order-preserving 64-bit key
    /// (`Kernels/GroupByExtrema.swift`) — the high word's extremes first, then the low word's among the
    /// rows that hold the winning high word. Unlike the plain atomic `min` / `max` this works for 64-bit
    /// types too, and unlike the segmented path it replaces there is no argsort of the key column.
    /// `segments` is accepted for source compatibility and only used by the segmented fallback.
    public func minMax<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil)
        throws -> (min: MetalArray<T>, max: MetalArray<T>) {
        try extrema(values)
    }

    /// The segmented `min_max` the atomic path replaced, kept for differential testing.
    func minMaxSegmented<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil)
        throws -> (min: MetalArray<T>, max: MetalArray<T>) {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let ctx = values.context
        let seg = try s ?? segments()
        let spec = ExtraAggregateSpec.of(T.self)
        let K = keyCount
        let mins = try MetalArrowBuffer.allocate(byteCount: K * 8, zeroed: false, context: ctx)
        let maxs = try MetalArrowBuffer.allocate(byteCount: K * 8, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(K, 1), context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "aggextra", source: spec.source,
                                        function: "gx_seg_minmax", type: spec.key)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            ExtraAggregates.bindSegments(enc, seg, values)
            enc.setBuffer(mins.mtl, offset: 0, index: 7)
            enc.setBuffer(maxs.mtl, offset: 0, index: 8)
            enc.setBuffer(validBytes.mtl, offset: 0, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: Swift.max(K, 1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        ctx.retainUntilFlush(seg.ord); ctx.retainUntilFlush(values)
        try ctx.syncPoint()
        return (try ExtraAggregates.decode(T.self, mins, validBytes, count: K, ctx),
                try ExtraAggregates.decode(T.self, maxs, validBytes, count: K, ctx))
    }

    /// `hash_min_max` as one Arrow `struct<min, max>` column, which is the shape Arrow's own
    /// `hash_min_max` returns.
    public func minMaxStruct<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil)
        throws -> MetalStructArray {
        let (lo, hi) = try minMax(values, segments: s)
        return try MetalStructArray(length: keyCount, nullCount: 0, validity: nil, names: ["min", "max"],
                                    children: [ExtraAggregates.erase(lo), ExtraAggregates.erase(hi)],
                                    context: values.context)
    }

    /// Arrow `hash_first_last` as one `struct<first, last>` column: the first and last non-null value of
    /// each key in row order. GPU, two group-by extremes over a row index that carries the values'
    /// validity, then two gathers.
    public func firstLast<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalStructArray {
        let f = try first(values), l = try last(values)
        return try MetalStructArray(length: keyCount, nullCount: 0, validity: nil, names: ["first", "last"],
                                    children: [ExtraAggregates.erase(f), ExtraAggregates.erase(l)],
                                    context: values.context)
    }

    /// Arrow `hash_one`: one arbitrary value per key.
    ///
    /// Arrow does not promise *which* row it picks, so this one is deliberate and reproducible: the
    /// **first non-null** value in row order, and null only when the key has no valid value. That is
    /// what pyarrow's `hash_one` returns on the same input, and it makes the answer the same run to run.
    ///
    /// GPU: a group-by minimum over a row index that carries the values' validity, then a gather.
    public func one<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<T> {
        try first(values)
    }

    /// `hash_one` in its other reading: the value of the lowest row of each key, null included.
    public func oneIncludingNull<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<T> {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let rows = try GroupByKeys.rowIndices(values.length, values.context)
        return try values.take(try min(rows))
    }

    /// Arrow `hash_list`: every value of each key, in row order, as one list per key.
    ///
    /// GPU: the stable sort by key that `segments()` already produces puts each key's rows together and
    /// in their original order, an exclusive scan of the per-key counts gives the list offsets, and one
    /// gather builds the child. Null values are kept as null list elements, which is what Arrow does;
    /// rows whose **key** is null or out of range are not in any list.
    public func list<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil) throws -> MetalListArray {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let seg = try s ?? segments()
        let counts = try count()
        let (offsets, total) = try ExtraAggregates.offsets(counts, keyCount: keyCount, ctx: values.context)
        let gather = try ExtraAggregates.gatherRuns(seg, offsets: offsets, total: total, keyCount: keyCount,
                                                    ctx: values.context)
        let child = try values.take(gather)
        return MetalListArray(length: keyCount, nullCount: 0, validity: nil, offsets: offsets,
                              values: ExtraAggregates.erase(child), context: values.context)
    }

    /// Arrow `hash_distinct`: the distinct non-null values of each key, ascending, as one list per key.
    ///
    /// GPU: the values are dictionary encoded, every row becomes the packed key `key * uniqueCount +
    /// code`, `unique()` collapses the repeats — which leaves the survivors already grouped by key and
    /// ordered by value inside each key — and one gather rebuilds the values.
    public func distinct<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalListArray {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let ctx = values.context
        let (codes, unique) = try values.dictionaryEncode()
        guard values.length > 0, unique.length > 0 else {
            return try ExtraAggregates.emptyLists(keyCount: keyCount, like: values, ctx: ctx)
        }
        let width = Int64(unique.length)
        let keys64 = try keys.cast(to: Int64.self)
        let codes64 = try codes.cast(to: Int64.self)
        let packed = try (try keys64.arithmetic(.mul, width)).arithmetic(.add, codes64)
        var distinctPacked = try packed.dropNull().unique()
        // Rows whose key falls outside [0, keyCount) pack to a value outside [0, keyCount * width);
        // dropping them here keeps the survivors in group order.
        let inRange = try (try distinctPacked.compare(.ge, 0)).and(try distinctPacked.compare(.lt, Int64(keyCount) * width))
        distinctPacked = try distinctPacked.filter(inRange)
        let groupOf = try distinctPacked.arithmetic(.div, width)
        let codeOf = try distinctPacked.arithmetic(.sub, try groupOf.arithmetic(.mul, width))
        let child = try unique.take(try codeOf.cast(to: Int32.self))
        let counts = try (try GroupBy<Int64>(keys: groupOf, keyCount: keyCount)).count()
        let (offsets, _) = try ExtraAggregates.offsets(counts, keyCount: keyCount, ctx: ctx)
        return MetalListArray(length: keyCount, nullCount: 0, validity: nil, offsets: offsets,
                              values: ExtraAggregates.erase(child), context: ctx)
    }

    /// Arrow `hash_product` per key, on the **GPU**: a segmented multiply reduction over the sorted key
    /// order. Integers accumulate in Int64 / UInt64 and wrap, as the scalar `product` does.
    /// A key with no valid value is null.
    public func productInt<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil)
        throws -> MetalArray<Int64> where T: FixedWidthInteger {
        let (raw, valid) = try segmentedProduct(values, segments: s)
        let out = try MetalArray<Int64>.allocate(length: keyCount, withValidity: true, context: values.context)
        withExtendedLifetime((raw, valid, out)) {
            let p = raw.typed(Int64.self), v = valid.typed(UInt8.self)
            let d = out.mutableValuePointer, bm = out.validity!.mutableTyped(UInt8.self)
            for k in 0..<keyCount where v[k] != 0 { d[k] = p[k]; Bitmap.set(bm, k) }
        }
        out.recomputeNullCount()
        return out
    }

    /// `hash_product` over a floating-point column, in Float64. Float32 values multiply in `float` and
    /// Float64 values through the software binary64 routine, both reassociated across the group.
    public func productFloat<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments? = nil)
        throws -> MetalArray<Double> {
        guard T.isFloatingPoint else {
            throw ArrowMetalError.unsupportedType("productFloat needs a floating-point column, got \(T.arrowFormat)")
        }
        let (raw, valid) = try segmentedProduct(values, segments: s)
        let isDouble = T.self == Double.self
        let out = try MetalArray<Double>.allocate(length: keyCount, withValidity: true, context: values.context)
        withExtendedLifetime((raw, valid, out)) {
            let v = valid.typed(UInt8.self)
            let d = out.mutableValuePointer, bm = out.validity!.mutableTyped(UInt8.self)
            for k in 0..<keyCount where v[k] != 0 {
                d[k] = isDouble ? Double(bitPattern: raw.typed(UInt64.self)[k]) : Double(raw.typed(Float.self)[k])
                Bitmap.set(bm, k)
            }
        }
        out.recomputeNullCount()
        return out
    }

    /// The raw segmented product accumulator per key plus a validity byte per key.
    func segmentedProduct<T: ArrowPrimitive>(_ values: MetalArray<T>, segments s: GroupSegments?)
        throws -> (MetalArrowBuffer, MetalArrowBuffer) {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let ctx = values.context
        let seg = try s ?? segments()
        let spec = ExtraAggregateSpec.of(T.self)
        let K = keyCount
        let out = try MetalArrowBuffer.allocate(byteCount: K * 8, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(K, 1), context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "aggextra", source: spec.source,
                                        function: "gx_seg_product", type: spec.key)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            ExtraAggregates.bindSegments(enc, seg, values)
            enc.setBuffer(out.mtl, offset: 0, index: 7)
            enc.setBuffer(validBytes.mtl, offset: 0, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: Swift.max(K, 1), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        ctx.retainUntilFlush(seg.ord); ctx.retainUntilFlush(values)
        try ctx.syncPoint()
        return (out, validBytes)
    }

    // MARK: order statistics

    /// Arrow `hash_quantile` with linear interpolation, computed **exactly**: the rows are sorted by
    /// (key, value) on the GPU and each key's answer is read at its interpolated position. `q` is
    /// clamped to [0, 1]. A key with no valid value is null.
    public func quantile<T: ArrowPrimitive>(_ values: MetalArray<T>, _ q: Double) throws -> MetalArray<Double> {
        let sortedOrder = try ExtraAggregates.SortedByGroup(self, values)
        return try sortedOrder.quantile(Swift.max(0, Swift.min(1, q)))
    }

    /// Arrow `hash_skew`: the third standardised central moment of each key.
    ///
    /// Arrow's default is the **biased** (population) form, `m3 / m2^1.5` with `m_k = sum((x - mean)^k) /
    /// n`. Pass `biased: false` for the sample-corrected form. A key with fewer than `minCount` valid
    /// values, or one whose values are all equal, is null.
    public func skew<T: ArrowPrimitive>(_ values: MetalArray<T>, biased: Bool = true, minCount: Int = 0,
                                        segments s: GroupSegments? = nil) throws -> MetalArray<Double> {
        try moments(values, biased: biased, minCount: minCount, wantSkew: true, segments: s)
    }

    /// Arrow `hash_kurtosis`: the **excess** kurtosis of each key, `m4 / m2^2 - 3` in its biased form
    /// (Arrow's default). Pass `biased: false` for the sample-corrected form.
    public func kurtosis<T: ArrowPrimitive>(_ values: MetalArray<T>, biased: Bool = true, minCount: Int = 0,
                                            segments s: GroupSegments? = nil) throws -> MetalArray<Double> {
        try moments(values, biased: biased, minCount: minCount, wantSkew: false, segments: s)
    }

    /// The shared two-pass body of `skew` and `kurtosis`.
    private func moments<T: ArrowPrimitive>(_ values: MetalArray<T>, biased: Bool, minCount: Int,
                                            wantSkew: Bool, segments s: GroupSegments?) throws -> MetalArray<Double> {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let ctx = values.context
        // Both passes in binary64 over the counting-sort order (`Kernels/GroupMoments.swift`), so the
        // deviations no longer round through Float32.
        let (sums, counts) = try centralMoments(values)
        try ctx.syncPoint()
        let K = keyCount
        let out = try MetalArray<Double>.allocate(length: K, withValidity: true, context: ctx)
        withExtendedLifetime((sums, counts, out)) {
            let s = sums.typed(UInt64.self), c = counts.typed(UInt32.self)
            let d = out.mutableValuePointer, bm = out.validity!.mutableTyped(UInt8.self)
            for k in 0..<K {
                let n = Int(c[k])
                guard n >= Swift.max(minCount, 1) else { continue }
                let m2 = Double(bitPattern: s[3 * k]) / Double(n)
                let m3 = Double(bitPattern: s[3 * k + 1]) / Double(n)
                let m4 = Double(bitPattern: s[3 * k + 2]) / Double(n)
                guard let v = ExtraAggregates.standardise(n: n, m2: m2, m3: m3, m4: m4,
                                                          wantSkew: wantSkew, biased: biased) else { continue }
                d[k] = v; Bitmap.set(bm, k)
            }
        }
        out.recomputeNullCount()
        return out
    }

    /// Arrow `hash_tdigest`: the t-digest estimate of `q` per key.
    ///
    /// GPU sort by (key, value), then one **host** pass per key merging the sorted values into centroids
    /// with the standard k1 scale function, exactly as `MetalArray.tdigest` does for a whole column.
    /// `bufferSize` is accepted for signature compatibility and has no effect: the values arrive fully
    /// sorted, so there is nothing to buffer.
    public func tdigest<T: ArrowPrimitive>(_ values: MetalArray<T>, _ q: Double, delta: Double = 100,
                                           bufferSize: Int = 500) throws -> MetalArray<Double> {
        let sortedOrder = try ExtraAggregates.SortedByGroup(self, values)
        return try sortedOrder.tdigest(q: q, delta: delta)
    }

    /// `sum` over any integer element type without the caller naming it, for the type-erased C layer.
    public func sumErasedInteger<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> {
        switch values {
        case let x as MetalArray<Int8>: return try sum(x)
        case let x as MetalArray<UInt8>: return try sum(x)
        case let x as MetalArray<Int16>: return try sum(x)
        case let x as MetalArray<UInt16>: return try sum(x)
        case let x as MetalArray<Int32>: return try sum(x)
        case let x as MetalArray<UInt32>: return try sum(x)
        case let x as MetalArray<Int64>: return try sum(x)
        case let x as MetalArray<UInt64>: return try sum(x)
        default: throw ArrowMetalError.unsupportedType("integer group-by sum over \(T.arrowFormat)")
        }
    }

    /// `hash_product` over any integer element type, for the type-erased C layer.
    public func productIntErased<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> {
        switch values {
        case let x as MetalArray<Int8>: return try productInt(x)
        case let x as MetalArray<UInt8>: return try productInt(x)
        case let x as MetalArray<Int16>: return try productInt(x)
        case let x as MetalArray<UInt16>: return try productInt(x)
        case let x as MetalArray<Int32>: return try productInt(x)
        case let x as MetalArray<UInt32>: return try productInt(x)
        case let x as MetalArray<Int64>: return try productInt(x)
        case let x as MetalArray<UInt64>: return try productInt(x)
        default: throw ArrowMetalError.unsupportedType("integer group-by product over \(T.arrowFormat)")
        }
    }

    // MARK: pivot

    /// Arrow `hash_pivot_wider` over a utf8 pivot-key column: one output field per name in `names`,
    /// holding the value of the row in that key whose pivot key equals the name (the lowest such row
    /// when there are several — Arrow raises on duplicates, this picks one).
    ///
    /// GPU: each name becomes a string equality mask, the mask is intersected with the values' validity,
    /// and the existing `hash_first` picks the survivor.
    public func pivotWider<T: ArrowPrimitive>(pivotKeys: MetalStringArray, values: MetalArray<T>,
                                              names: [String]) throws -> MetalStructArray {
        guard pivotKeys.length == keys.length, values.length == keys.length else {
            throw ArrowMetalError.lengthMismatch(keys.length, Swift.min(pivotKeys.length, values.length))
        }
        var children: [AnyMetalArray] = []
        for name in names {
            let mask = try pivotKeys.matches(.equals, name)
            children.append(ExtraAggregates.erase(try first(try ExtraAggregates.masked(values, by: mask))))
        }
        return try MetalStructArray(length: keyCount, nullCount: 0, validity: nil, names: names,
                                    children: children, context: values.context)
    }
}

// MARK: - scalar skew / kurtosis / tdigest

extension MetalArray {

    /// Arrow `skew`: the third standardised central moment, `m3 / m2^1.5`, in Arrow's default **biased**
    /// (population) form. `biased: false` gives the sample-corrected form. Nil when there are fewer than
    /// `minCount` valid values, when fewer than three remain for the unbiased form, or when every value
    /// is the same (the denominator is then zero).
    ///
    /// Two GPU passes, exactly like `variance`: `sum()` for the mean, then one kernel for the second,
    /// third and fourth central moments about it.
    public func skew(biased: Bool = true, minCount: Int = 0) throws -> Double? {
        guard let m = try centralMoments() else { return nil }
        guard m.count >= Swift.max(minCount, 1) else { return nil }
        return ExtraAggregates.standardise(n: m.count, m2: m.m2, m3: m.m3, m4: m.m4, wantSkew: true, biased: biased)
    }

    /// Arrow `kurtosis`: the **excess** kurtosis `m4 / m2^2 - 3`, biased by default as Arrow's is.
    public func kurtosis(biased: Bool = true, minCount: Int = 0) throws -> Double? {
        guard let m = try centralMoments() else { return nil }
        guard m.count >= Swift.max(minCount, 1) else { return nil }
        return ExtraAggregates.standardise(n: m.count, m2: m.m2, m3: m.m3, m4: m.m4, wantSkew: false, biased: biased)
    }

    /// The second, third and fourth central moments of the non-null values, and how many contributed.
    func centralMoments() throws -> (m2: Double, m3: Double, m4: Double, count: Int)? {
        let valid = validCount
        guard valid > 0, let s = try sum() else { return nil }
        let mean = s.asDouble / Double(valid)
        guard mean.isFinite else { return nil }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let spec = ExtraAggregateSpec.of(T.self)
        let groups = Aggregates.groupCount(n)
        let wide = spec.moment == .double
        let partials = try MetalArrowBuffer.allocate(byteCount: groups * 3 * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: groups * 4, zeroed: false, context: ctx)
        var params = AggMeanParams()
        switch spec.moment {
        case .integer:
            let floored = mean.rounded(.down)
            params.ipart = Int64(clampedTo: floored)
            params.hi = Float(mean - floored)
        case .float:
            params.hi = Float(mean)
            params.lo = Float(mean - Double(Float(mean)))
        case .double:
            params.bits = mean.bitPattern
        }
        let pso = try Dispatch.pipeline(ctx, family: "aggextra", source: spec.source,
                                        function: "gx_moment34", type: spec.key)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            Aggregates.bindValues(enc, self)
            enc.setBytes(&params, length: MemoryLayout<AggMeanParams>.size, index: 4)
            enc.setBuffer(partials.mtl, offset: 0, index: 5)
            enc.setBuffer(counts.mtl, offset: 0, index: 6)
            Aggregates.dispatch(enc, groups: groups)
        }
        try ctx.syncPoint()
        return withExtendedLifetime((partials, counts)) { () -> (Double, Double, Double, Int)? in
            let c = counts.typed(UInt32.self)
            var t2 = 0.0, t3 = 0.0, t4 = 0.0, total = 0
            for g in 0..<groups where c[g] > 0 {
                total += Int(c[g])
                if wide {
                    let p = partials.typed(UInt64.self)
                    t2 += Double(bitPattern: p[3 * g])
                    t3 += Double(bitPattern: p[3 * g + 1])
                    t4 += Double(bitPattern: p[3 * g + 2])
                } else {
                    let p = partials.typed(Float.self)
                    // Each accumulator is a compensated (sum, correction) pair.
                    t2 += Double(p[6 * g]) + Double(p[6 * g + 1])
                    t3 += Double(p[6 * g + 2]) + Double(p[6 * g + 3])
                    t4 += Double(p[6 * g + 4]) + Double(p[6 * g + 5])
                }
            }
            guard total > 0 else { return nil }
            return (t2 / Double(total), t3 / Double(total), t4 / Double(total), total)
        }
    }

    /// Arrow `tdigest`: the t-digest estimate of quantile `q`.
    ///
    /// The values are sorted on the GPU and merged into centroids in one host pass with the standard k1
    /// scale function `delta * (asin(2q - 1) / pi + 0.5)`. Because the input is already fully sorted,
    /// `bufferSize` has nothing to buffer and is accepted only for signature compatibility with Arrow;
    /// the digest this builds is the one a single global merge produces, which is at least as accurate
    /// as the incrementally buffered one Arrow builds. See the class comment for the agreement measured
    /// against `pyarrow.compute.tdigest`.
    public func tdigest(_ q: Double, delta: Double = 100, bufferSize: Int = 500) throws -> Double? {
        try tdigest([q], delta: delta, bufferSize: bufferSize).first ?? nil
    }

    /// Several quantiles from the same digest, which costs one sort rather than one per quantile.
    ///
    /// The digest of a sorted stream of unit weights holds one value per centroid, whatever the
    /// compression (`Kernels/TDigestGPU.swift` derives it), so it is the sorted column itself and the
    /// quantile is read out of it directly rather than by walking ten million values through
    /// `TDigest.add`. Bit-identical to that walk; `ARROWMETAL_TDIGEST_WALK=1` runs it instead.
    public func tdigest(_ qs: [Double], delta: Double = 100, bufferSize: Int = 500) throws -> [Double?] {
        let m = validCount
        guard m > 0 else { return qs.map { _ in nil } }
        // The digest only ever sees the valid values, and sorting a column that carries a validity
        // bitmap costs three times sorting one that does not (117 ms against 39 ms at 10M float64), so
        // the nulls are compacted out first — one filter pass — rather than sorted to the end.
        let sortedValues = try TDigestGPU.strippedOfNulls(self).sorted()
        let valid = Swift.min(m, sortedValues.length)
        return withExtendedLifetime(sortedValues) { () -> [Double?] in
            let p = sortedValues.valuePointer
            if TDigestGPU.walkEveryValue {
                var d = TDigest(delta: delta)
                for i in 0..<valid { d.add(p[i].asDouble) }
                return qs.map { d.quantile($0) }
            }
            // NaN is skipped by `add` without being counted, and the sort puts every NaN after every
            // value, so the digest is built from the leading `count` values.
            let count = MetalArray<T>.nonNaNPrefix(sortedValues, valid)
            return qs.map { TDigest.sortedQuantile($0, count: count) { p[$0].asDouble } }
        }
    }
}

// MARK: - plumbing

/// Shared helpers for the aggregates above.
enum ExtraAggregates {

    /// Buffers 0 to 6 of every segmented kernel here: bounds, order, values, validity, count, flag.
    static func bindSegments<V: ArrowPrimitive>(_ enc: MTLComputeCommandEncoder, _ seg: GroupSegments,
                                                _ values: MetalArray<V>) {
        enc.setBuffer(seg.segStart.mtl, offset: seg.segStart.offset, index: 0)
        enc.setBuffer(seg.segEnd.mtl, offset: seg.segEnd.offset, index: 1)
        let o = seg.ord.length > 0 ? seg.ord.values : seg.segStart
        enc.setBuffer(o.mtl, offset: o.offset, index: 2)
        enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 3)
        let vv = values.validity ?? values.values
        enc.setBuffer(vv.mtl, offset: vv.offset, index: 4)
        Dispatch.setLength(enc, seg.keyCount, nil, index: 5)
        Dispatch.setUInt(enc, values.validity == nil ? 0 : 1, index: 6)
    }

    /// One min/max output buffer plus its validity bytes as a typed array.
    static func decode<T: ArrowPrimitive>(_: T.Type, _ raw: MetalArrowBuffer, _ validBytes: MetalArrowBuffer,
                                          count K: Int, _ ctx: MetalContext) throws -> MetalArray<T> {
        let out = try MetalArray<T>.allocate(length: K, withValidity: true, context: ctx)
        withExtendedLifetime((raw, validBytes, out)) {
            let v = validBytes.typed(UInt8.self)
            let d = out.mutableValuePointer, bm = out.validity!.mutableTyped(UInt8.self)
            for k in 0..<K where v[k] != 0 {
                d[k] = Aggregates.decodeKey(T.self, raw, k)
                Bitmap.set(bm, k)
            }
        }
        out.recomputeNullCount()
        return out
    }

    /// Int32 list offsets from per-group counts: a GPU inclusive scan plus a one-element shift.
    /// Returns the offsets buffer (K + 1 entries) and the total child length.
    static func offsets(_ counts: MetalArray<Int64>, keyCount K: Int, ctx: MetalContext) throws -> (MetalArrowBuffer, Int) {
        let off = try MetalArrowBuffer.allocate(byteCount: (K + 1) * 4, context: ctx)
        guard K > 0 else { return (off, 0) }
        let cum = try counts.cumulativeSum()
        let pso = try ctx.pipeline(source: AggregatesExtraSource.common, function: "gx_offsets",
                                   cacheKey: "aggextra/common/gx_offsets")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(cum.values.mtl, offset: cum.values.offset, index: 0)
            Dispatch.setLength(enc, K, nil, index: 1)
            enc.setBuffer(off.mtl, offset: off.offset, index: 2)
            Dispatch.dispatch1D(enc, pso, count: K + 1)
        }
        ctx.retainUntilFlush(cum)
        try ctx.syncPoint()
        let total = withExtendedLifetime(off) { Int(off.typed(Int32.self)[K]) }
        return (off, total)
    }

    /// Concatenates each group's run of the sorted order into one gather index.
    static func gatherRuns(_ seg: GroupSegments, offsets: MetalArrowBuffer, total: Int, keyCount K: Int,
                           ctx: MetalContext) throws -> MetalArray<Int32> {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1) * 4, context: ctx)
        guard total > 0 else { return MetalArray<Int32>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }
        let pso = try ctx.pipeline(source: AggregatesExtraSource.common, function: "gx_gather_runs",
                                   cacheKey: "aggextra/common/gx_gather_runs")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(seg.segStart.mtl, offset: seg.segStart.offset, index: 0)
            enc.setBuffer(seg.segEnd.mtl, offset: seg.segEnd.offset, index: 1)
            let o = seg.ord.length > 0 ? seg.ord.values : seg.segStart
            enc.setBuffer(o.mtl, offset: o.offset, index: 2)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 3)
            Dispatch.setLength(enc, K, nil, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: K, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        ctx.retainUntilFlush(seg.ord)
        try ctx.syncPoint()
        return MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// A list column of `keyCount` empty lists over an empty child of the same type.
    static func emptyLists<T: ArrowPrimitive>(keyCount K: Int, like: MetalArray<T>, ctx: MetalContext) throws -> MetalListArray {
        let off = try MetalArrowBuffer.allocate(byteCount: (K + 1) * 4, context: ctx)
        return MetalListArray(length: K, nullCount: 0, validity: nil, offsets: off,
                              values: erase(try MetalArray<T>([T](), context: ctx)), context: ctx)
    }

    /// The values with their validity intersected with a boolean mask (false and null both mask out).
    static func masked<T: ArrowPrimitive>(_ values: MetalArray<T>, by mask: MetalBooleanArray) throws -> MetalArray<T> {
        let ctx = values.context
        let n = values.length
        let effective = try BitmapOps.combineValidity(ctx, mask.values, mask.validity, bits: n)
        let combined = try BitmapOps.combineValidity(ctx, values.validity, effective, bits: n)
        try ctx.syncPoint()
        let out = MetalArray<T>(length: n, nullCount: 0, validity: combined, values: values.values, context: ctx)
        out.recomputeNullCount()
        return out
    }

    /// `skew` / `kurtosis` from the central moments, in Arrow's biased or sample-corrected form.
    static func standardise(n: Int, m2: Double, m3: Double, m4: Double, wantSkew: Bool, biased: Bool) -> Double? {
        guard m2 > 0 else { return nil }
        if wantSkew {
            let g1 = m3 / (m2 * m2.squareRoot())
            guard g1.isFinite else { return nil }
            if biased { return g1 }
            guard n > 2 else { return nil }
            let d = Double(n)
            return g1 * ((d * (d - 1)).squareRoot() / (d - 2))
        }
        let g2 = m4 / (m2 * m2) - 3
        guard g2.isFinite else { return nil }
        if biased { return g2 }
        guard n > 3 else { return nil }
        let d = Double(n)
        return ((d - 1) / ((d - 2) * (d - 3))) * ((d + 1) * g2 + 6)
    }

    /// A `MetalArray<T>` as an `AnyMetalArray`, without going through the C layer's `wrap`.
    static func erase<T: ArrowPrimitive>(_ a: MetalArray<T>) -> AnyMetalArray {
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
        default: return .int64(try! MetalArray<Int64>([Int64](), context: a.context))
        }
    }

    /// The rows of a `GroupBy` sorted by (group, value), the shape every per-group order statistic needs.
    ///
    /// Two stable GPU radix argsorts: by value first (nulls last), then by key. The second sort being
    /// stable is what leaves each group's rows in ascending value order with its nulls at the end.
    struct SortedByGroup<K: ArrowIndex, T: ArrowPrimitive> {
        let gb: GroupBy<K>
        let values: MetalArray<T>
        /// Row indices, grouped by key and ascending by value inside each key.
        let ord: MetalArray<Int32>
        /// First sorted position of each key; a key with no row keeps zero and is reported null.
        let segStart: MetalArrowBuffer
        /// Non-null values per key.
        let counts: MetalArray<Int64>
        let ctx: MetalContext

        init(_ gb: GroupBy<K>, _ values: MetalArray<T>) throws {
            guard values.length == gb.keys.length else {
                throw ArrowMetalError.lengthMismatch(gb.keys.length, values.length)
            }
            let ctx = values.context
            let n = values.length
            let m = n - gb.keys.nullCount
            let K = gb.keyCount
            let segStart = try MetalArrowBuffer.allocate(byteCount: K * 4, context: ctx)
            let segEnd = try MetalArrowBuffer.allocate(byteCount: K * 4, context: ctx)
            var order = try MetalArray<Int32>([Int32](), context: ctx)
            if m > 0 {
                let byValue = try values.argsort()
                let keysInValueOrder = try gb.keys.take(byValue)
                order = try byValue.take(try keysInValueOrder.argsort()).slice(offset: 0, length: m)
                let sortedKeys = try (try gb.keys.take(order)).cast(to: Int32.self)
                let pso = try ctx.pipeline(source: AggregatesExtraSource.common, function: "gx_bounds",
                                           cacheKey: "aggextra/common/gx_bounds")
                try ctx.run { enc in
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(sortedKeys.values.mtl, offset: sortedKeys.values.offset, index: 0)
                    Dispatch.setLength(enc, m, nil, index: 1)
                    Dispatch.setUInt(enc, K, index: 2)
                    enc.setBuffer(segStart.mtl, offset: segStart.offset, index: 3)
                    enc.setBuffer(segEnd.mtl, offset: segEnd.offset, index: 4)
                    Dispatch.dispatch1D(enc, pso, count: m)
                }
                ctx.retainUntilFlush(sortedKeys)
                try ctx.syncPoint()
            }
            self.gb = gb
            self.values = values
            self.ord = order
            self.segStart = segStart
            self.counts = try gb.countValid(values)
            self.ctx = ctx
        }

        /// The interpolated quantile of every key, exactly.
        ///
        /// The two bracketing positions are computed on the host in `Double` — `q * (count - 1)` in
        /// Float32 would already be off by whole rows at 50 million — and the GPU gathers the two
        /// values behind them.
        func quantile(_ q: Double) throws -> MetalArray<Double> {
            let K = gb.keyCount
            let spec = ExtraAggregateSpec.of(T.self)
            let width = Swift.max(T.byteWidth, 4)
            let lows = try MetalArrowBuffer.allocate(byteCount: Swift.max(K, 1) * width, context: ctx)
            let highs = try MetalArrowBuffer.allocate(byteCount: Swift.max(K, 1) * width, context: ctx)
            let loIdx = try MetalArrowBuffer.allocate(byteCount: Swift.max(K, 1) * 4, context: ctx)
            let hiIdx = try MetalArrowBuffer.allocate(byteCount: Swift.max(K, 1) * 4, context: ctx)
            let wanted = try MetalArrowBuffer.allocate(byteCount: Swift.max(K, 1), context: ctx)
            var weights = [Double](repeating: 0, count: K)
            withExtendedLifetime((counts, loIdx, hiIdx, wanted)) {
                let c = counts.valuePointer
                let lo = loIdx.mutableTyped(UInt32.self), hi = hiIdx.mutableTyped(UInt32.self)
                let w = wanted.mutableTyped(UInt8.self)
                for k in 0..<K {
                    let n = Int(c[k])
                    guard n > 0 else { continue }
                    let pos = q * Double(n - 1)
                    lo[k] = UInt32(pos.rounded(.down))
                    hi[k] = UInt32(pos.rounded(.up))
                    weights[k] = pos - pos.rounded(.down)
                    w[k] = 1
                }
            }
            if ord.length > 0 {
                let pso = try Dispatch.pipeline(ctx, family: "aggextra", source: spec.source,
                                                function: "gx_seg_pick", type: spec.key)
                try ctx.run { enc in
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(segStart.mtl, offset: segStart.offset, index: 0)
                    enc.setBuffer(ord.values.mtl, offset: ord.values.offset, index: 1)
                    enc.setBuffer(values.values.mtl, offset: values.values.offset, index: 2)
                    enc.setBuffer(loIdx.mtl, offset: 0, index: 3)
                    enc.setBuffer(hiIdx.mtl, offset: 0, index: 4)
                    enc.setBuffer(wanted.mtl, offset: 0, index: 5)
                    Dispatch.setLength(enc, K, nil, index: 6)
                    enc.setBuffer(lows.mtl, offset: 0, index: 7)
                    enc.setBuffer(highs.mtl, offset: 0, index: 8)
                    Dispatch.dispatch1D(enc, pso, count: K)
                }
                ctx.retainUntilFlush(ord); ctx.retainUntilFlush(values); ctx.retainUntilFlush(counts)
                try ctx.syncPoint()
            }
            let out = try MetalArray<Double>.allocate(length: K, withValidity: true, context: ctx)
            withExtendedLifetime((lows, highs, wanted, out)) {
                let v = wanted.typed(UInt8.self)
                let lo = lows.typed(T.self), hi = highs.typed(T.self)
                let d = out.mutableValuePointer, bm = out.validity!.mutableTyped(UInt8.self)
                for k in 0..<K where v[k] != 0 {
                    let a = lo[k].asDouble, b = hi[k].asDouble
                    d[k] = a + (b - a) * weights[k]
                    Bitmap.set(bm, k)
                }
            }
            out.recomputeNullCount()
            return out
        }

        /// The t-digest estimate of `q` per key: one host merge over each key's sorted run.
        func tdigest(q: Double, delta: Double) throws -> MetalArray<Double> {
            let K = gb.keyCount
            let out = try MetalArray<Double>.allocate(length: K, withValidity: true, context: ctx)
            try ctx.syncPoint()
            withExtendedLifetime((ord, values, counts, segStart, out)) {
                let starts = segStart.typed(UInt32.self)
                let c = counts.valuePointer
                let o = ord.valuePointer
                let v = values.valuePointer
                let d = out.mutableValuePointer, bm = out.validity!.mutableTyped(UInt8.self)
                for k in 0..<K {
                    let n = Int(c[k])
                    guard n > 0 else { continue }
                    var digest = TDigest(delta: delta)
                    let s = Int(starts[k])
                    for j in 0..<n { digest.add(v[Int(o[s + j])].asDouble) }
                    guard let est = digest.quantile(q) else { continue }
                    d[k] = est; Bitmap.set(bm, k)
                }
            }
            out.recomputeNullCount()
            return out
        }
    }
}

/// A t-digest built by merging an **already sorted** stream of values, with the standard k1 scale
/// function. This is Arrow's merging algorithm minus the buffering: Arrow sorts 500 values at a time and
/// merges each buffer into the digest, while here the whole column arrives sorted from the GPU, so one
/// merge does it. The resulting digest is at least as accurate as the buffered one.
struct TDigest {
    struct Centroid { var mean: Double; var weight: Double }

    let delta: Double
    private(set) var centroids: [Centroid] = []
    private var totalWeight = 0.0
    private var weightSoFar = 0.0
    private var weightLimit = -1.0
    private var minimum = Double.infinity
    private var maximum = -Double.infinity

    init(delta: Double = 100) { self.delta = Swift.max(delta, 1) }

    /// The k1 scale function and its inverse.
    private func k(_ q: Double) -> Double { delta * (asin(2 * q - 1) / Double.pi + 0.5) }
    private func q(_ kk: Double) -> Double { (sin((kk / delta - 0.5) * Double.pi) + 1) / 2 }

    /// `quantile(target)` of the digest a sorted stream of `count` unit weights builds, read straight
    /// out of the sorted values instead of out of a materialised centroid array.
    ///
    /// Every centroid of that digest holds exactly one value — see the derivation in
    /// `Kernels/TDigestGPU.swift`: `weightLimit` is scaled by the weight seen *so far*, and the scale
    /// function's inverse is bounded by 1, so the limit never reaches the next value's weight. With
    /// unit weights the walk in `quantile` above lands on centroid `ceil(index) - 1` and the
    /// interpolation reads at most two neighbouring values, so this is that function transcribed term
    /// for term with `centroids[i].mean` replaced by `value(i)` and every weight by 1.
    ///
    /// `value` reads the sorted, non-null, non-NaN values; `count` is how many there are.
    static func sortedQuantile(_ target: Double, count: Int, value: (Int) -> Double) -> Double? {
        guard count > 0 else { return nil }
        let totalWeight = Double(count)
        let minimum = value(0), maximum = value(count - 1)
        if target <= 0 { return minimum }
        if target >= 1 { return maximum }
        let index = target * totalWeight
        if index <= 1 { return minimum }
        if index >= totalWeight - 1 { return maximum }
        // The walk accumulates one unit per centroid and stops at the first `index <= weightSum`.
        var ci = Int((index - 1).rounded(.up))
        ci = Swift.min(Swift.max(ci, 0), count - 1)
        var diff = index + 0.5 - Double(ci + 1)
        if abs(diff) < 0.5 { return value(ci) }
        var left = ci, right = ci
        if diff > 0 {
            if right == count - 1 { return lerp(value(right), maximum, diff / 0.5) }
            right += 1
        } else {
            if left == 0 { return lerp(minimum, value(0), index / 0.5) }
            left -= 1
            diff += 1
        }
        return lerp(value(left), value(right), diff)
    }

    private static func lerp(_ a: Double, _ b: Double, _ t: Double) -> Double { a + (b - a) * t }

    /// Adds one value of the sorted stream. `totalWeight` is not known up front, so the limit is scaled
    /// by the weight seen so far — which for a sorted single pass is the same digest a two-pass merge
    /// would build once every value has been seen.
    mutating func add(_ x: Double) {
        guard x.isNaN == false else { return }
        minimum = Swift.min(minimum, x)
        maximum = Swift.max(maximum, x)
        totalWeight += 1
        merge(Centroid(mean: x, weight: 1))
    }

    private mutating func merge(_ c: Centroid) {
        let weight = weightSoFar + c.weight
        if weight <= weightLimit, var last = centroids.last {
            let w = last.weight + c.weight
            last.mean += (c.mean - last.mean) * c.weight / w
            last.weight = w
            centroids[centroids.count - 1] = last
        } else {
            let quantile = totalWeight > 0 ? weightSoFar / totalWeight : 0
            let next = totalWeight * q(k(quantile) + 1)
            weightLimit = next <= weightLimit ? totalWeight : next
            centroids.append(c)
        }
        weightSoFar = weight
    }

    /// Arrow's `TDigest::Quantile`: locate the centroid holding the position, then interpolate between
    /// it and its neighbour (or the recorded minimum / maximum at the ends).
    func quantile(_ target: Double) -> Double? {
        guard !centroids.isEmpty, totalWeight > 0 else { return nil }
        if target <= 0 { return minimum }
        if target >= 1 { return maximum }
        let index = target * totalWeight
        if index <= 1 { return minimum }
        if index >= totalWeight - 1 { return maximum }
        var ci = 0
        var weightSum = 0.0
        while ci < centroids.count {
            weightSum += centroids[ci].weight
            if index <= weightSum { break }
            ci += 1
        }
        if ci == centroids.count { ci = centroids.count - 1 }
        var diff = index + centroids[ci].weight / 2 - weightSum
        if centroids[ci].weight == 1, abs(diff) < 0.5 { return centroids[ci].mean }
        var left = ci, right = ci
        if diff > 0 {
            if right == centroids.count - 1 {
                let half = centroids[right].weight / 2
                return half > 0 ? lerp(centroids[right].mean, maximum, diff / half) : maximum
            }
            right += 1
        } else {
            if left == 0 {
                let half = centroids[0].weight / 2
                return half > 0 ? lerp(minimum, centroids[0].mean, index / half) : minimum
            }
            left -= 1
            diff += centroids[left].weight / 2 + centroids[right].weight / 2
        }
        let span = centroids[left].weight / 2 + centroids[right].weight / 2
        guard span > 0 else { return centroids[left].mean }
        return lerp(centroids[left].mean, centroids[right].mean, diff / span)
    }

    private func lerp(_ a: Double, _ b: Double, _ t: Double) -> Double { a + (b - a) * t }
}
