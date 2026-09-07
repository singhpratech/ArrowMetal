import Foundation

// Streaming quantiles by a mergeable digest.
//
// An exact quantile needs the whole column ordered, which an out-of-core query cannot afford. A
// digest keeps a bounded set of weighted centroids instead: each says "about w values sit near v".
// Two digests merge by concatenating their centroids, ordering them and compressing back to the
// budget, which is associative and order independent — exactly what a streaming aggregate needs.
//
// Per batch the GPU does the work: sort the (non-null) values with the existing radix argsort, then
// gather `compression` sample points with one `take`. The sample positions are *not* uniform in rank;
// they follow the inverse of the t-digest k1 scale function, which packs samples into the tails,
// where a uniform sample is worst. The merge then re-compresses under the same scale function, so a
// centroid near q = 0.99 keeps far less weight than one near q = 0.5.
//
// Error: a centroid covering rank interval [r0, r1) answers any quantile inside it with rank error
// at most (r1 - r0) / 2. With the k1 scale and compression C, that is about n / C at the median and
// n * pi / (2 C^2) near the extremes — so p99 is roughly C times more accurate than the median.

/// A mergeable digest of a distribution: centroids `(value, weight)` in ascending value order.
public struct StreamDigest: Sendable {
    public private(set) var values: [Double] = []
    public private(set) var weights: [Double] = []
    /// Centroid budget. Larger is more accurate and more expensive to merge.
    public let compression: Int
    public private(set) var totalWeight: Double = 0
    public private(set) var minimum = Double.infinity
    public private(set) var maximum = -Double.infinity

    public init(compression: Int = 1000) { self.compression = Swift.max(16, compression) }

    init(compression: Int, values: [Double], weights: [Double], min: Double, max: Double) {
        self.compression = compression
        self.values = values
        self.weights = weights
        self.totalWeight = weights.reduce(0, +)
        self.minimum = min
        self.maximum = max
    }

    public var isEmpty: Bool { totalWeight == 0 }

    /// Merges `other` in, then compresses back to the budget.
    public mutating func merge(_ other: StreamDigest) {
        guard !other.isEmpty else { return }
        if isEmpty {
            values = other.values; weights = other.weights
            totalWeight = other.totalWeight
            minimum = other.minimum; maximum = other.maximum
            compress()
            return
        }
        // Merge two ascending centroid lists.
        var v: [Double] = [], w: [Double] = []
        v.reserveCapacity(values.count + other.values.count)
        w.reserveCapacity(values.count + other.values.count)
        var i = 0, j = 0
        while i < values.count || j < other.values.count {
            if j >= other.values.count || (i < values.count && values[i] <= other.values[j]) {
                v.append(values[i]); w.append(weights[i]); i += 1
            } else {
                v.append(other.values[j]); w.append(other.weights[j]); j += 1
            }
        }
        values = v; weights = w
        totalWeight += other.totalWeight
        minimum = Swift.min(minimum, other.minimum)
        maximum = Swift.max(maximum, other.maximum)
        compress()
    }

    /// t-digest k1 scale: k(q) = (C / 2pi) * asin(2q - 1). Centroids may absorb weight while the
    /// scale distance they span stays under 1, which allows fat clusters in the middle and thin ones
    /// at the tails.
    private func k1(_ q: Double) -> Double {
        let x = Swift.min(1, Swift.max(-1, 2 * q - 1))
        return Double(compression) / (2 * Double.pi) * asin(x)
    }

    private mutating func compress() {
        guard values.count > compression, totalWeight > 0 else { return }
        var outV: [Double] = [], outW: [Double] = []
        outV.reserveCapacity(compression + 1)
        outW.reserveCapacity(compression + 1)
        var carriedV = values[0] * weights[0]
        var carriedW = weights[0]
        var qStart = 0.0
        for i in 1..<values.count {
            let qEnd = (qStart * totalWeight + carriedW + weights[i]) / totalWeight
            if k1(qEnd) - k1(qStart) <= 1 {
                carriedV += values[i] * weights[i]
                carriedW += weights[i]
            } else {
                outV.append(carriedV / carriedW)
                outW.append(carriedW)
                qStart += carriedW / totalWeight
                carriedV = values[i] * weights[i]
                carriedW = weights[i]
            }
        }
        outV.append(carriedV / carriedW)
        outW.append(carriedW)
        values = outV
        weights = outW
    }

    /// The estimated `q` quantile (linear interpolation between centroid midpoints).
    public func quantile(_ q: Double) -> Double? {
        guard !isEmpty, !values.isEmpty else { return nil }
        if q <= 0 { return minimum }
        if q >= 1 { return maximum }
        let target = q * totalWeight
        var cumulative = 0.0
        for i in 0..<values.count {
            let mid = cumulative + weights[i] / 2
            if target <= mid {
                if i == 0 {
                    // Between the smallest value and the first centroid.
                    let t = mid == 0 ? 0 : target / mid
                    return minimum + t * (values[0] - minimum)
                }
                let prevMid = cumulative - weights[i - 1] / 2
                let span = mid - prevMid
                let t = span == 0 ? 0 : (target - prevMid) / span
                return values[i - 1] + t * (values[i] - values[i - 1])
            }
            cumulative += weights[i]
        }
        let lastMid = cumulative - weights[values.count - 1] / 2
        let span = totalWeight - lastMid
        let t = span == 0 ? 0 : (target - lastMid) / span
        return values[values.count - 1] + t * (maximum - values[values.count - 1])
    }
}

/// Builds a digest of one column's non-null values on the GPU.
///
/// One radix argsort plus one gather. Only `compression` values ever reach the host.
public func gpuDigest(_ column: AnyMetalArray, compression: Int = 1000) throws -> StreamDigest {
    let dbl = try toDouble(column)
    let clean = try dbl.dropNull()
    let n = clean.length
    guard n > 0 else { return StreamDigest(compression: compression) }
    let sorted = try clean.sorted()
    let c = Swift.min(Swift.max(16, compression), n)
    // Sample positions follow the inverse of the k1 scale, so the tails are sampled densely.
    var idx: [Int32] = []
    var weights: [Double] = []
    idx.reserveCapacity(c)
    weights.reserveCapacity(c)
    var previous = 0
    for j in 0..<c {
        let qHigh = (1 + sin(Double.pi * (Double(j + 1) / Double(c) - 0.5))) / 2
        var end = Int((qHigh * Double(n)).rounded())
        if j == c - 1 { end = n }
        end = Swift.min(n, Swift.max(previous + 1, end))
        idx.append(Int32(Swift.min(n - 1, (previous + end) / 2)))
        weights.append(Double(end - previous))
        previous = end
        if previous >= n { break }
    }
    let gathered = try sorted.take(try MetalArray<Int32>(idx, context: clean.context))
    // Everything above is recorded into whatever batch is open; run it before reading the samples.
    try clean.context.syncPoint()
    var values: [Double] = []
    values.reserveCapacity(idx.count)
    gathered.withValues { buf in for v in buf { values.append(v) } }
    let lo = sorted[0] ?? values.first ?? 0
    let hi = sorted[n - 1] ?? values.last ?? 0
    // Values from a sorted array are already ascending; keep only the ones with weight.
    var v: [Double] = [], w: [Double] = []
    for (i, wt) in weights.enumerated() where wt > 0 && i < values.count {
        v.append(values[i]); w.append(wt)
    }
    return StreamDigest(compression: compression, values: v, weights: w, min: lo, max: hi)
}

/// Streaming quantiles: a per-batch GPU digest merged into one running digest.
public final class StreamQuantileOperator: StreamOperator {
    public let column: String
    public let quantiles: [Double]
    public let compression: Int
    public let filter: Expr?
    private var digest: StreamDigest

    public init(column: String, quantiles: [Double], compression: Int = 1000, filter: Expr? = nil) {
        self.column = column
        self.quantiles = quantiles
        self.compression = compression
        self.filter = filter
        self.digest = StreamDigest(compression: compression)
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        var col: AnyMetalArray
        if let f = filter {
            let r = try streamFilterProject(batch, filter: f, projections: [("v", .column(column))],
                                            context: batch.firstContext ?? .shared)
            guard let c = r["v"] else { return nil }
            col = c
        } else {
            guard let c = batch[column] else {
                throw ArrowMetalError.invalidArrowArray("no column named \(column)")
            }
            col = c
        }
        guard col.length > 0 else { return nil }
        return try gpuDigest(col, compression: compression)
    }

    /// Two centroid lists merged on the host; no kernel, so no command buffer.
    public var mergeUsesGPU: Bool { false }

    public func merge(_ partial: Any) throws {
        guard let d = partial as? StreamDigest else { return }
        digest.merge(d)
    }

    public func finish() throws -> StreamResult {
        var r = StreamResult()
        for q in quantiles {
            r.scalarNames.append("q\(q)")
            r.scalars.append(digest.quantile(q).map { ExprScalar.double($0) } ?? .null)
        }
        return r
    }

    /// The merged digest, for callers that want more quantiles without a second pass.
    public var mergedDigest: StreamDigest { digest }
}
