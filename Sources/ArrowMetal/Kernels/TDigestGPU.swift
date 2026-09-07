import Foundation
import Metal

// `tdigest` without the ten-million-step host walk.
//
// `MetalArray.tdigest` sorted on the GPU and then fed every value to `TDigest.add` one at a time —
// 557 ms at 10M rows against pyarrow's 206 ms, and 0.14 GB/s, which is what a host loop over device
// memory looks like. The plan for this page was a GPU segmented sum over the centroid runs, so that
// only the centroids reach the host. Working out where the runs actually fall showed something better,
// and something worth writing down:
//
// **This digest never merges two values into one centroid.** Feeding a sorted stream of unit weights,
// `TDigest.merge` folds value `j` into the open centroid only when `j + 1 <= weightLimit`, and it sets
//
//     weightLimit = totalWeight * q(k(weightSoFar / totalWeight) + 1)
//
// when a centroid opens, where `totalWeight` is the weight seen *so far*, `j + 1`, not the column's
// final weight. `q` is `(sin(...) + 1) / 2`, whose range is `[0, 1]`, so `weightLimit <= j + 1 < j + 2`
// and the next value always opens a centroid of its own. (The `next <= weightLimit` guard sets the
// limit to `totalWeight` instead, which is `j + 1` and just as short.) That holds for every
// compression and every column length — Arrow avoids it by scaling against the final weight, which a
// single streaming pass does not know.
//
// So the digest of a sorted column *is* the sorted column, one unit centroid per value, and
// `TDigest.quantile` over unit centroids collapses to a closed form: the walk that finds the centroid
// holding position `index = q * n` lands on `ceil(index) - 1`, and the interpolation that follows
// reads at most two neighbouring values. `TDigest.sortedQuantile` is that transcription, term for
// term, and `TDigestTests.testSortedQuantileMatchesTheWalk` holds it against the walk itself.
//
// The result is bit-identical to what the walk returned, which matters: `tdigest` is compared against
// `pc.tdigest` and against the exact quantile by the differential harness and the function table, and
// this digest happens to be exact. Making it a real ≤1000-centroid sketch would move every answer by
// far more than those oracles allow, and is a semantics change, not a performance fix.
//
// What is left is the sort, which was always the honest cost of the operation.

extension MetalArray {

    /// How many of the sorted values are non-NaN. `TDigest.add` skips a NaN without counting it, and
    /// ArrowMetal's sort puts every NaN after every value (and every null after those), so `isNaN` is
    /// false then true over `0 ..< m` and one binary search finds the boundary. An integer column has
    /// none to find.
    static func nonNaNPrefix(_ sortedValues: MetalArray<T>, _ m: Int) -> Int {
        guard T.isFloatingPoint, m > 0 else { return m }
        return withExtendedLifetime(sortedValues) { () -> Int in
            let p = sortedValues.valuePointer
            guard p[m - 1].asDouble.isNaN else { return m }
            var lo = 0, hi = m - 1                     // p[hi] is NaN, so the answer is in [lo, hi]
            while lo < hi {
                let mid = (lo + hi) / 2
                if p[mid].asDouble.isNaN { hi = mid } else { lo = mid + 1 }
            }
            return lo
        }
    }
}

enum TDigestGPU {
    /// Set `ARROWMETAL_TDIGEST_WALK=1` to build the digest by walking every value through
    /// `TDigest.add`, which is what `tdigest` did before. Both give the same answer; this is how the
    /// before/after numbers were measured in one binary.
    static let walkEveryValue = ProcessInfo.processInfo.environment["ARROWMETAL_TDIGEST_WALK"] != nil

    /// The column with its nulls compacted away and no validity bitmap left behind.
    ///
    /// Sorting a nullable column is three times the work of sorting the same values without a bitmap
    /// (117 ms against 39 ms at 10M float64), and the digest never looks at the nulls — `sorted()` puts
    /// them past `validCount` and the walk stops there. Dropping them first is one filter pass, and it
    /// leaves the very same sequence of values: `filter` is stable, so equal keys keep their relative
    /// order and the sorted result is identical element for element.
    static func strippedOfNulls<T: ArrowPrimitive>(_ a: MetalArray<T>) throws -> MetalArray<T> {
        guard a.nullCount > 0 else { return a }
        let d = try a.dropNull()
        guard d.validity != nil, d.nullCount == 0 else { return d }
        return MetalArray<T>(length: d.length, nullCount: 0, validity: nil, values: d.values,
                             context: d.context)
    }
}
