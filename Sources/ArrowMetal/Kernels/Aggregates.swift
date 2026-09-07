import Foundation
import Metal

// The scalar aggregates Arrow defines beyond sum / min / max / mean / count, on the GPU.
//
// | function | how it runs |
// |---|---|
// | `product` | one GPU pass, per-threadgroup partial products, host combine |
// | `variance` / `stddev` | two GPU passes (sum, then squared deviations from that mean), host combine |
// | `min_max` | one GPU pass producing both partials |
// | `index` | one GPU pass, device-wide atomic minimum over the matching rows |
// | `first` / `last` | one GPU pass over the validity bitmap (atomic min/max index), one host read |
// | `any` / `all` | one GPU pass, word-wise popcounts of `values & validity` and of `validity` |
// | `quantile` / `approximate_median` | GPU radix select on the order-preserving key: exact, not approximate |
// | `mode` | GPU `value_counts` plus a host argmax |
// | `count_distinct` | GPU `unique().length` |
//
// `tdigest` is **out of scope**: it is an approximate sketch whose merge step is inherently sequential
// per digest, and the exact `quantile` here covers the same question on the data sizes this package
// targets. There is no plan to add it.
//
// Precision. Integer `product` accumulates in Int64 / UInt64 and wraps, as Arrow's does. Float32
// products accumulate per thread in `float` and are combined in `double`, so they reassociate and every
// multiplication rounds in float: over thousands of factors expect a relative error around 1e-5.
// Float64 products run the software binary64 multiply on the GPU (`DoubleMath`), correctly rounded per
// operation but still reassociated across threads. Variance and standard deviation are computed in two
// passes: the mean comes from `sum()`, and the squared deviations are accumulated in compensated float
// pairs (Neumaier) for integer and Float32 columns, or in software binary64 for Float64 columns, then
// combined on the host in `Double`. Expect a relative error of about 1e-7 for Float32 and integer
// columns and about 1e-15 for Float64 columns. Integer deviations subtract the integer part of the mean
// in 64-bit before converting to float, so a column of large integers keeps its precision; the
// exception is a UInt64 column with values above `Int64.max`, where that subtraction wraps.

extension MetalArray {

    // MARK: - product

    /// Arrow `product` of the non-null values, or nil when there is no valid value.
    ///
    /// Integers accumulate in Int64 / UInt64 and wrap on overflow, matching Arrow. Float32 accumulates
    /// in `float` per thread and combines in `double`; Float64 multiplies through the software binary64
    /// routine on the GPU.
    public func product() throws -> SumResult? {
        guard validCount > 0 else { return nil }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let spec = AggregateSpec.of(T.self)
        let groups = Aggregates.groupCount(n)
        let partials = try MetalArrowBuffer.allocate(byteCount: groups * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: groups * 4, zeroed: false, context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "aggregate", source: spec.source, function: "agg_product", type: spec.key)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            Aggregates.bindValues(enc, self)
            enc.setBuffer(partials.mtl, offset: 0, index: 4)
            enc.setBuffer(counts.mtl, offset: 0, index: 5)
            Aggregates.dispatch(enc, groups: groups)
        }
        try ctx.syncPoint()
        return withExtendedLifetime((partials, counts)) {
            if T.self == Double.self {
                let p = partials.typed(UInt64.self)
                var acc = 1.0
                for g in 0..<groups { acc *= Double(bitPattern: p[g]) }
                return .float(acc)
            } else if T.isFloatingPoint {
                let p = partials.typed(Float.self)
                var acc = 1.0
                for g in 0..<groups { acc *= Double(p[g]) }
                return .float(acc)
            } else if T.minValue < 0 as T {
                let p = partials.typed(Int64.self)
                var acc: Int64 = 1
                for g in 0..<groups { acc &*= p[g] }
                return .int(acc)
            } else {
                let p = partials.typed(UInt64.self)
                var acc: UInt64 = 1
                for g in 0..<groups { acc &*= p[g] }
                return .uint(acc)
            }
        }
    }

    // MARK: - variance / stddev

    /// Arrow `variance`. `ddof` is the delta degrees of freedom: 0 (the default) is the population
    /// variance, 1 the sample variance. Nil when fewer than `ddof + 1` valid values remain.
    ///
    /// Two passes: `sum()` for the mean, then a GPU pass over the squared deviations from it.
    public func variance(ddof: Int = 0) throws -> Double? {
        guard let (sumSquares, count) = try squaredDeviations(), count > ddof else { return nil }
        return sumSquares / Double(count - ddof)
    }

    /// Population standard deviation (`ddof: 1` gives the sample one). Nil when there is nothing to measure.
    public func stddev(ddof: Int = 0) throws -> Double? {
        guard let v = try variance(ddof: ddof) else { return nil }
        return v.squareRoot()
    }

    /// The sum of squared deviations from the mean and the number of values that contributed.
    private func squaredDeviations() throws -> (Double, Int)? {
        let valid = validCount
        guard valid > 0, let s = try sum() else { return nil }
        let mean = s.asDouble / Double(valid)
        // A NaN or infinite mean (a NaN in a Float64 column, or an overflowing float sum) has no variance.
        guard mean.isFinite else { return nil }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let spec = AggregateSpec.of(T.self)
        let groups = Aggregates.groupCount(n)
        let partials = try MetalArrowBuffer.allocate(byteCount: groups * 8, zeroed: false, context: ctx)
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
        let pso = try Dispatch.pipeline(ctx, family: "aggregate", source: spec.source, function: "agg_moment", type: spec.key)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            Aggregates.bindValues(enc, self)
            enc.setBytes(&params, length: MemoryLayout<AggMeanParams>.size, index: 4)
            enc.setBuffer(partials.mtl, offset: 0, index: 5)
            enc.setBuffer(counts.mtl, offset: 0, index: 6)
            Aggregates.dispatch(enc, groups: groups)
        }
        try ctx.syncPoint()
        return withExtendedLifetime((partials, counts)) {
            let c = counts.typed(UInt32.self)
            var total = 0.0, contributed = 0
            if spec.moment == .double {
                let p = partials.typed(UInt64.self)
                for g in 0..<groups where c[g] > 0 { total += Double(bitPattern: p[g]); contributed += Int(c[g]) }
            } else {
                let p = partials.typed(Float.self)
                for g in 0..<groups where c[g] > 0 {
                    total += Double(p[2 * g]) + Double(p[2 * g + 1])
                    contributed += Int(c[g])
                }
            }
            return (total, contributed)
        }
    }

    // MARK: - min_max

    /// Arrow `min_max` in a single pass: one kernel produces both partials, so the values are read once.
    /// Nil when every value is null (or NaN, which is skipped as `min` / `max` do).
    public func minMax() throws -> (min: T, max: T)? {
        guard validCount > 0 else { return nil }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let spec = AggregateSpec.of(T.self)
        let groups = Aggregates.groupCount(n)
        let mins = try MetalArrowBuffer.allocate(byteCount: groups * 8, zeroed: false, context: ctx)
        let maxs = try MetalArrowBuffer.allocate(byteCount: groups * 8, zeroed: false, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: groups * 4, zeroed: false, context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "aggregate", source: spec.source, function: "agg_minmax", type: spec.key)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            Aggregates.bindValues(enc, self)
            enc.setBuffer(mins.mtl, offset: 0, index: 4)
            enc.setBuffer(maxs.mtl, offset: 0, index: 5)
            enc.setBuffer(counts.mtl, offset: 0, index: 6)
            Aggregates.dispatch(enc, groups: groups)
        }
        try ctx.syncPoint()
        return withExtendedLifetime((mins, maxs, counts)) { () -> (min: T, max: T)? in
            let c = counts.typed(UInt32.self)
            var lo = T.maxValue, hi = T.minValue, any = false
            for g in 0..<groups where c[g] > 0 {
                any = true
                lo = Swift.min(lo, Aggregates.decodeKey(T.self, mins, g))
                hi = Swift.max(hi, Aggregates.decodeKey(T.self, maxs, g))
            }
            return any ? (lo, hi) : nil
        }
    }

    // MARK: - quantile / median / mode / count_distinct

    /// Arrow `quantile` with linear interpolation, computed exactly. `q` is clamped to [0, 1].
    ///
    /// Only the one or two values that bracket the interpolated position are needed, so this is a GPU radix
    /// select on the order-preserving key (`Kernels/RadixSelectValue.swift`) — two passes over the column —
    /// rather than a full sort. Small columns keep the sort, which is already cheap there.
    ///
    /// Nulls are skipped. A NaN is not a null: it sorts after `+inf` and therefore drags a high quantile
    /// with it, exactly as sorting the column and indexing it would.
    public func quantile(_ q: Double) throws -> Double? {
        let m = validCount
        guard m > 0 else { return nil }
        let position = Swift.max(0.0, Swift.min(1.0, q)) * Double(m - 1)
        let low = Int(position.rounded(.down)), high = Int(position.rounded(.up))
        if let (a, b) = try quantileBounds(low: low, high: high) {
            return low == high ? a : a + (b - a) * (position - Double(low))
        }
        let sortedValues = try sorted()
        return withExtendedLifetime(sortedValues) {
            let p = sortedValues.valuePointer
            let a = p[low].asDouble, b = p[high].asDouble
            if low == high { return a }
            return a + (b - a) * (position - Double(low))
        }
    }

    /// The values at ranks `low` and `high` (adjacent, or equal) among the non-null values, by radix select.
    /// Nil when the selection path does not apply and the caller should sort instead.
    private func quantileBounds(low: Int, high: Int) throws -> (Double, Double)? {
        guard length >= 1 << 12 else { return nil }        // sorting a small column is already cheap
        guard let found = try radixSelectKey(rank: low, largest: false) else { return nil }
        let a = TopK.value(fromKey: found.key, largest: false, T.self).asDouble
        if low == high { return (a, a) }
        // The two ranks are adjacent, so one search settles both unless `low` was the last row of the bin.
        var next = found.next
        if next == nil { next = try radixSelectKey(rank: high, largest: false)?.key }
        guard let nk = next else { return nil }
        return (a, TopK.value(fromKey: nk, largest: false, T.self).asDouble)
    }

    /// Arrow `approximate_median`, computed exactly (`quantile(0.5)`): this package selects the middle rank
    /// on the GPU instead of sketching, so there is nothing approximate about the answer.
    public func approximateMedian() throws -> Double? { try quantile(0.5) }

    /// Arrow `mode`: the most common non-null value and how often it occurs. Ties go to the smallest
    /// value, as Arrow does. Nil when every value is null.
    ///
    /// GPU `value_counts` (sort, mark runs, scan) plus a host argmax over the distinct values.
    public func mode() throws -> (value: T, count: Int64)? {
        let (values, counts) = try valueCounts()
        guard values.length > 0 else { return nil }
        return withExtendedLifetime((values, counts)) { () -> (value: T, count: Int64)? in
            let v = values.valuePointer, c = counts.valuePointer
            var best = 0
            for i in 1..<values.length where c[i] > c[best] { best = i }
            return (v[best], c[best])
        }
    }

    /// Arrow `count_distinct` over the non-null values (`mode = "only_valid"`).
    ///
    /// Above `1 << 16` rows this is the GPU hash table's occupied-slot count (`Kernels/HashTable.swift`)
    /// — the distinct values are never gathered and never ordered, because a count does not need them.
    /// Smaller inputs go through `unique()`.
    public func countDistinct() throws -> Int {
        guard Self.prefersHashTable(rows: length) else { return try unique().length }
        guard let (groups, _) = try hashGroups() else { return 0 }
        return groups.groupCount
    }

    // MARK: - first / last / index

    /// Arrow `first`: the first value, skipping nulls unless `skipNulls` is false (in which case a null
    /// first row gives nil). Nil for an empty array or one with no valid value.
    public func first(skipNulls: Bool = true) throws -> T? {
        guard length > 0 else { return nil }
        guard skipNulls, rawValidity != nil else { return self[0] }
        guard let bounds = try validBounds() else { return nil }
        return withExtendedLifetime(self) { valuePointer[bounds.first] }
    }

    /// Arrow `last`, with the same null handling as `first`.
    public func last(skipNulls: Bool = true) throws -> T? {
        let n = length
        guard n > 0 else { return nil }
        guard skipNulls, rawValidity != nil else { return self[n - 1] }
        guard let bounds = try validBounds() else { return nil }
        return withExtendedLifetime(self) { valuePointer[bounds.last] }
    }

    /// First and last row with a validity bit set.
    ///
    /// Read on the host from Metal shared memory, 64 bitmap bits at a time inwards from each end, so the
    /// cost is the distance to the first (last) valid row — O(1) for the usual mostly-valid column, and no
    /// GPU dispatch at all. Only a column whose first `Aggregates.hostScanLimit` bits are all null (and
    /// which is large enough for a full GPU pass to be worth its ~150 us dispatch floor) falls through to
    /// the atomic min / max kernel below.
    func validBounds() throws -> (first: Int, last: Int)? {
        guard let v = rawValidity else { return length > 0 ? (0, length - 1) : nil }
        let ctx = context
        let n = length
        guard n > 0 else { return nil }
        let off = offset
        let limit = n > Aggregates.gpuScanFloor ? Aggregates.hostScanLimit : Int.max
        let host: (first: Int, last: Int)?? = withExtendedLifetime(v) { () -> (first: Int, last: Int)?? in
            let p = v.typed(UInt8.self)
            guard let f = Bitmap.firstSet(p, from: off, bits: Swift.min(n, limit)) else {
                if limit < n { return nil }        // inconclusive: escalate to the GPU
                return .some(nil)                  // scanned the whole column: no valid row
            }
            guard let l = Bitmap.lastSet(p, from: off, bits: n) else { return .some(nil) }
            return .some((f, l))
        }
        if let answer = host { return answer }
        guard validCount > 0 else { return nil }
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: 8, zeroed: false, context: ctx)
        withExtendedLifetime(out) {
            let p = out.mutableTyped(UInt32.self)
            p[0] = UInt32.max      // smallest valid index
            p[1] = 0               // largest valid index, stored as index + 1
        }
        let pso = try Dispatch.pipeline(ctx, family: "aggregate", source: AggregatesSource.common,
                                        function: "agg_valid_bounds", type: "common")
        guard let vn = validity else { return (0, n - 1) }   // normalised: the kernel sees no offset
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(vn.mtl, offset: vn.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer(out.mtl, offset: out.offset, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset + 4, index: 3)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        try ctx.syncPoint()
        return withExtendedLifetime(out) { () -> (first: Int, last: Int)? in
            let p = out.typed(UInt32.self)
            guard p[0] != UInt32.max, p[1] > 0 else { return nil }
            return (Int(p[0]), Int(p[1]) - 1)
        }
    }

    /// Arrow `index`: the first row that equals `value`, or -1 when it is absent.
    ///
    /// GPU: every matching row atomically lowers a single device-wide minimum. Equality is Arrow value
    /// equality for floats (`-0.0` equals `0.0`, and NaN equals nothing, including itself).
    public func index(of value: T) throws -> Int64 {
        let ctx = context
        let n = length
        guard n > 0 else { return -1 }
        try Dispatch.checkLength(n)
        let spec = AggregateSpec.of(T.self)
        let out = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: false, context: ctx)
        withExtendedLifetime(out) { out.mutableTyped(UInt32.self)[0] = UInt32.max }
        let pso = try Dispatch.pipeline(ctx, family: "aggregate", source: spec.source, function: "agg_index", type: spec.key)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            Aggregates.bindValues(enc, self)
            if let d = value as? Double { Dispatch.setScalar(enc, Int64(bitPattern: d.bitPattern), index: 4) }
            else { Dispatch.setScalar(enc, value, index: 4) }
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        try ctx.syncPoint()
        let found = withExtendedLifetime(out) { out.typed(UInt32.self)[0] }
        return found == UInt32.max ? -1 : Int64(found)
    }
}

// MARK: - boolean any / all on the GPU

extension MetalBooleanArray {
    /// Number of true (and valid) values and number of valid values, in one GPU pass over the bitmap
    /// words. The host-side `trueCount` property computes the same first number on the CPU.
    public func trueAndValidCounts() throws -> (trueCount: Int, validCount: Int) {
        let n = length
        guard n > 0 else { return (0, 0) }
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: 8, context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "aggregate", source: AggregatesSource.common,
                                        function: "agg_bool_counts", type: "common")
        let vld = validity ?? values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 2)
            Dispatch.setLength(enc, n, nil, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset + 4, index: 5)
            Dispatch.dispatch1D(enc, pso, count: BitmapOps.words(bits: n))
        }
        try ctx.syncPoint()
        return withExtendedLifetime(out) {
            let p = out.typed(UInt32.self)
            return (Int(p[0]), Int(p[1]))
        }
    }

    /// Arrow `any` with `skip_nulls`: true when at least one valid value is true.
    ///
    /// A word-wise host scan of `values & validity` in Metal shared memory that stops at the first true
    /// bit, so the answer is usually two loads and no GPU work at all — this is what the CPU libraries do,
    /// and a full GPU pass cannot beat a short circuit. A large column whose first
    /// `Aggregates.hostScanLimit` bits give no answer escalates to the counting kernel.
    public func anyTrue() throws -> Bool {
        let n = length
        guard n > 0 else { return false }
        let limit = n > Aggregates.gpuScanFloor ? Aggregates.hostScanLimit : Int.max
        let host = withExtendedLifetime(self) { () -> Bool? in
            Bitmap.anySet(rawValues.typed(UInt8.self), rawValidity?.typed(UInt8.self),
                          from: offset, bits: n, limit: limit)
        }
        if let host { return host }
        return try trueAndValidCounts().trueCount > 0
    }

    /// Arrow `all` with `skip_nulls`: true when every valid value is true, and true for an empty or
    /// all-null array, matching Arrow's default `min_count = 0`.
    ///
    /// The mirror of `anyTrue`: a host scan that stops at the first word holding a valid false.
    public func allTrue() throws -> Bool {
        let n = length
        guard n > 0 else { return true }
        let limit = n > Aggregates.gpuScanFloor ? Aggregates.hostScanLimit : Int.max
        let host = withExtendedLifetime(self) { () -> Bool? in
            Bitmap.allSet(rawValues.typed(UInt8.self), rawValidity?.typed(UInt8.self),
                          from: offset, bits: n, limit: limit)
        }
        if let host { return host }
        let (t, v) = try trueAndValidCounts()
        return t == v
    }
}

// MARK: - grouped aggregates (hash_* over dense keys)

// The grouped forms of the aggregates above, over the same dense keys `GroupBy` already takes (the codes
// a dictionary encoding produces). Each one is built from GPU primitives that already exist:
//
// | function | how it runs |
// |---|---|
// | `hash_first` / `hash_last` | the sort-free grouped extremes over a row index array, then one `take` |
// | `hash_any` / `hash_all` | group-by max / min over the unpacked boolean bytes — all GPU |
// | `hash_variance` / `hash_stddev` | two GPU passes in binary64 over the counting-sort order |
// | `hash_count_distinct` | GPU hash **set** over the (key, value) pair, then a histogram of the occupied slots |
// | `hash_product` | one host pass over the key and value buffers (there is no 64-bit atomic multiply) |
// | `hash_approximate_median` | **not implemented** — see `approximateMedian` below |

extension GroupBy {
    /// Arrow `hash_first`: the first non-null value of each key, in row order. Keys with no valid value
    /// are null. GPU: a group-by minimum over the row indices, then a `take`.
    public func first<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<T> {
        try values.take(try rowIndex(values, wantFirst: true))
    }

    /// Arrow `hash_last`: the last non-null value of each key. GPU, as `first`.
    public func last<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<T> {
        try values.take(try rowIndex(values, wantFirst: false))
    }

    /// The first (or last) row index per key among the rows whose value is non-null, or null for a key
    /// with no such row. The row indices carry the values' validity bitmap, so nulls are skipped.
    private func rowIndex<T: ArrowPrimitive>(_ values: MetalArray<T>, wantFirst: Bool) throws -> MetalArray<Int32> {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        let iota = try MetalArray<Int32>.iota(values.length, context: values.context)
        let masked = MetalArray<Int32>(length: values.length, nullCount: values.nullCount,
                                       validity: values.validity, values: iota.values, context: values.context)
        // The fused sort-free extremes: one atomic pass and a GPU finalize, so a ten-million-group
        // `first` never walks the groups on the host.
        let e = try extrema(masked)
        return wantFirst ? e.min : e.max
    }

    /// Arrow `hash_any` over a boolean column: true for a key with at least one true value, null for a
    /// key with no valid value. GPU: the bits become bytes and a group-by maximum does the rest.
    public func any(_ values: MetalBooleanArray) throws -> MetalBooleanArray {
        try MetalBooleanArray.fromUInt8Array(try max(try values.toUInt8Array()))
    }

    /// Arrow `hash_all` over a boolean column: true when every valid value of the key is true.
    public func all(_ values: MetalBooleanArray) throws -> MetalBooleanArray {
        try MetalBooleanArray.fromUInt8Array(try min(try values.toUInt8Array()))
    }

    /// Arrow `hash_product` per key, wrapping in Int64 exactly as the scalar `product` does.
    ///
    /// This runs on the **GPU** (`Kernels/AggregatesExtra.swift`): Metal has no 64-bit atomic multiply,
    /// so the multiplication is segmented instead of atomic — the keys are argsorted once and one
    /// threadgroup multiplies one key's run. Keys outside `[0, keyCount)` and null keys are skipped, as
    /// everywhere else in `GroupBy`; a key with no valid value is null.
    public func product<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> where T: FixedWidthInteger {
        try productInt(values)
    }

    /// Arrow `hash_variance` per key (`ddof` 0 for the population variance, 1 for the sample one).
    ///
    /// GPU, in the same two-pass shape as the scalar form, but in **true binary64**
    /// (`Kernels/GroupMoments.swift`): the counting sort by group id puts each group's rows together,
    /// one segmented pass sums the values exactly for the per-group mean, and a second sums the
    /// deviations and their squares about it. Every addition, multiplication and division goes through
    /// the software binary64 routines, and the shift correction `sum d^2 - (sum d)^2 / n` removes the
    /// error in the mean itself, so a Float64 column lands within about 1e-14 relative of Arrow's own
    /// answer rather than the 1e-5 the Float32 deviations used to cost.
    public func variance<T: ArrowPrimitive>(_ values: MetalArray<T>, ddof: Int = 0) throws -> MetalArray<Double> {
        try varianceDouble(values, ddof: ddof)
    }

    /// Arrow `hash_stddev` per key: the square root of `variance`.
    public func stddev<T: ArrowPrimitive>(_ values: MetalArray<T>, ddof: Int = 0) throws -> MetalArray<Double> {
        let v = try variance(values, ddof: ddof)
        let out = try MetalArray<Double>.allocate(length: keyCount, withValidity: true, context: v.context)
        withExtendedLifetime((v, out)) {
            let s = v.valuePointer
            let vb = v.validity?.typed(UInt8.self)
            let d = out.mutableValuePointer, valid = out.validity!.mutableTyped(UInt8.self)
            for k in 0..<keyCount where vb == nil || Bitmap.isSet(vb!, k) {
                d[k] = s[k].squareRoot(); Bitmap.set(valid, k)
            }
        }
        out.recomputeNullCount()
        return out
    }

    /// The mean of each key as a Float32 array (0 where a key has no valid value).
    private func meanPerKey<T: ArrowPrimitive>(_ values: MetalArray<T>, counts: MetalArray<Int64>) throws -> MetalArray<Float> {
        let sums: [Double]
        if let f = values as? MetalArray<Float> {
            let s = try sumFloat(f)
            sums = (0..<keyCount).map { s.isValid($0) ? s.valuePointer[$0] : 0 }
        } else if let d = values as? MetalArray<Double> {
            throw ArrowMetalError.unsupportedType("group-by variance over Float64 values (cast to Float32 first); \(d.length) rows")
        } else {
            let s = try sumErased(values)
            sums = (0..<keyCount).map { s.isValid($0) ? Double(s.valuePointer[$0]) : 0 }
        }
        var means = [Float](repeating: 0, count: keyCount)
        for k in 0..<keyCount where counts.valuePointer[k] > 0 {
            means[k] = Float(sums[k] / Double(counts.valuePointer[k]))
        }
        return try MetalArray<Float>(means, context: values.context)
    }

    /// `sum` over any integer element type, without the caller having to name it.
    private func sumErased<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> {
        switch values {
        case let x as MetalArray<Int8>: return try sum(x)
        case let x as MetalArray<UInt8>: return try sum(x)
        case let x as MetalArray<Int16>: return try sum(x)
        case let x as MetalArray<UInt16>: return try sum(x)
        case let x as MetalArray<Int32>: return try sum(x)
        case let x as MetalArray<UInt32>: return try sum(x)
        case let x as MetalArray<Int64>: return try sum(x)
        case let x as MetalArray<UInt64>: return try sum(x)
        default: throw ArrowMetalError.unsupportedType("grouped statistics over \(T.arrowFormat)")
        }
    }

    /// Arrow `hash_count_distinct`: distinct non-null values per key.
    ///
    /// GPU: a hash **set** over the (key, value) pair (`Kernels/GroupCountDistinct.swift`) — one insert
    /// pass over the rows, then one pass over the table's occupied slots, each of which is one distinct
    /// pair and increments its group's count. Rows with a null key, a null value or a key outside
    /// `[0, keyCount)` are never inserted.
    ///
    /// `ARROWMETAL_NO_HASH=1` falls back to the packed-key path this replaced: dictionary-encode the
    /// values, pack each row into `key * uniqueCount + code`, `unique()` the packed column and count
    /// what is left per key. Both give the same answer — the set's value key is the normalisation
    /// `dictionaryEncode()` already applies, so all NaNs are one value and `-0.0` is `0.0` either way —
    /// and the fallback is how the before/after numbers were measured in one binary.
    public func countDistinct<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Int64> {
        guard values.length == keys.length else { throw ArrowMetalError.lengthMismatch(keys.length, values.length) }
        if let hashed = try countDistinctHashed(values) { return hashed }
        let ctx = values.context
        let (codes, unique) = try values.dictionaryEncode()
        let width = Int64(Swift.max(unique.length, 1))
        guard values.length > 0, unique.length > 0 else {
            return MetalArray<Int64>(length: keyCount, nullCount: 0, validity: nil,
                                     values: try MetalArrowBuffer.allocate(byteCount: keyCount * 8, context: ctx),
                                     context: ctx)
        }
        // Pack (key, code) into one int64. Rows whose key or code is null carry a null through the
        // arithmetic and are dropped before the distinct pass.
        let keys64 = try keys.cast(to: Int64.self)
        let codes64 = try codes.cast(to: Int64.self)
        let packed = try (try keys64.arithmetic(.mul, width)).arithmetic(.add, codes64)
        let distinct = try packed.dropNull().unique()
        let distinctKeys = try distinct.arithmetic(.div, width)
        let counts = try (try GroupBy<Int64>(keys: distinctKeys, keyCount: keyCount)).count()
        return counts
    }

    /// Arrow `hash_approximate_median` per key, computed **exactly** — `quantile(values, 0.5)`.
    ///
    /// The segmented sort it needs now exists (`Kernels/AggregatesExtra.swift`): two stable GPU radix
    /// argsorts put the rows in (key, value) order, and a per-key kernel reads the two values that
    /// bracket the median position. A key with no valid value is null.
    public func approximateMedian<T: ArrowPrimitive>(_ values: MetalArray<T>) throws -> MetalArray<Double> {
        try quantile(values, 0.5)
    }
}

// MARK: - plumbing

/// Mean parameters for the second variance pass; the layout matches `AggMeanParams` in the MSL source.
struct AggMeanParams {
    var ipart: Int64 = 0
    var bits: UInt64 = 0
    var hi: Float = 0
    var lo: Float = 0
}

enum Aggregates {
    /// How many bits a short-circuiting host scan (`first`, `last`, `any`, `all`) reads before it gives up
    /// and lets a full GPU pass answer instead. 1 Mibit is 128 KB, well inside L2, and takes roughly 10 us
    /// to scan — about a fifteenth of the GPU's dispatch floor, so escalating is never the wrong call.
    static let hostScanLimit = 1 << 20
    /// Below this length a full host scan is cheaper than any dispatch, so the scan never escalates.
    static let gpuScanFloor = 8 << 20

    /// Enough threadgroups to saturate the GPU while keeping the host combine trivial.
    static func groupCount(_ n: Int) -> Int {
        Swift.max(1, Swift.min(2048, (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize))
    }

    static func dispatch(_ enc: MTLComputeCommandEncoder, groups: Int) {
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
    }

    /// Values, validity, length and a validity flag: buffers 0 to 3 of every aggregate kernel.
    static func bindValues<T: ArrowPrimitive>(_ enc: MTLComputeCommandEncoder, _ a: MetalArray<T>) {
        enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
        let v = a.validity ?? a.values
        enc.setBuffer(v.mtl, offset: v.offset, index: 1)
        Dispatch.setLength(enc, a.length, nil, index: 2)
        Dispatch.setUInt(enc, a.validity == nil ? 0 : 1, index: 3)
    }

    /// Turns one min/max partial back into an element value (the inverse of the kernel's `ag_key`).
    static func decodeKey<T: ArrowPrimitive>(_: T.Type, _ buffer: MetalArrowBuffer, _ g: Int) -> T {
        if T.self == Double.self { return Dispatch.doubleFromKey(buffer.typed(Int64.self)[g]) as! T }
        if T.isFloatingPoint { return T(buffer.typed(Float.self)[g]) }
        if T.minValue < 0 as T { return T(truncatingIfNeededInt64: buffer.typed(Int64.self)[g]) }
        return T(truncatingIfNeededUInt64: buffer.typed(UInt64.self)[g])
    }
}

/// The generated MSL for one element type, plus the cache key it is compiled under.
struct AggregateSpec {
    let source: String
    let key: String
    let moment: AggregatesSource.MomentMode

    static func of<T: ArrowPrimitive>(_: T.Type) -> AggregateSpec {
        if T.self == Double.self {
            // Float64 travels as raw 64-bit patterns; every arithmetic step is a software binary64 routine.
            return AggregateSpec(
                source: AggregatesSource.source(
                    T: "long", ACC: "ulong", KEYACC: "long",
                    productInit: "0x3FF0000000000000ul", productMul: "d_mul(a, b)", load: "(ulong)vals[i]",
                    include: "!d_isnan((long)vals[i])", key: "d_key((long)vals[i])",
                    minInit: "LONG_MAX", maxInit: "LONG_MIN",
                    equal: "(!d_isnan((long)a) && !d_isnan((long)b) && d_key((long)a) == d_key((long)b))",
                    moment: .double, extraPrelude: DoubleMath.msl),
                key: "g", moment: .double)
        }
        if T.isFloatingPoint {
            return AggregateSpec(
                source: AggregatesSource.source(
                    T: "float", ACC: "float", KEYACC: "float",
                    productInit: "1.0f", productMul: "a * b", load: "(float)vals[i]",
                    include: "!isnan(vals[i])", key: "vals[i]",
                    minInit: "INFINITY", maxInit: "-INFINITY", equal: "a == b", moment: .float),
                key: "f", moment: .float)
        }
        let signed = T.minValue < 0 as T
        return AggregateSpec(
            source: AggregatesSource.source(
                T: T.mslType, ACC: signed ? "long" : "ulong", KEYACC: signed ? "long" : "ulong",
                // Signed multiplication is done on the unsigned representation so overflow wraps rather
                // than being undefined, which is what Arrow's `product` specifies.
                productInit: signed ? "1L" : "1ul",
                productMul: signed ? "(long)((ulong)a * (ulong)b)" : "a * b",
                load: signed ? "(long)vals[i]" : "(ulong)vals[i]",
                include: "true", key: signed ? "(long)vals[i]" : "(ulong)vals[i]",
                minInit: signed ? "LONG_MAX" : "ULONG_MAX", maxInit: signed ? "LONG_MIN" : "0ul",
                equal: "a == b", moment: .integer),
            key: T.arrowFormat, moment: .integer)
    }
}

extension Int64 {
    /// Saturating conversion from a Double that may be out of Int64's range.
    init(clampedTo d: Double) {
        if d.isNaN { self = 0 }
        else if d <= -9.223372036854775e18 { self = .min }
        else if d >= 9.223372036854775e18 { self = .max }
        else { self = Int64(d) }
    }
}
