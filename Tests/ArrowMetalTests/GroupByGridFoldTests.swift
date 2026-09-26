import XCTest
@testable import ArrowMetal

/// The per-group kernels (one 256-thread threadgroup per group) at and past 2^24 groups.
///
/// The GPU holds a grid dimension's thread count in 32 bits, so a `(K, 1, 1)` grid of 256-thread
/// threadgroups wraps at K = 2^24 and runs only `K mod 2^24` threadgroups, with no error. Before
/// `Dispatch.perGroup` folded such grids into rows, a Float64 sum or mean over 2^24 groups came back
/// all null, and over 2^24 + 1 groups every group but the first. These tests pin the answers at the
/// boundary and, with the fold forced on at small sizes, every kernel that reads its group from the
/// folded grid.
final class GroupByGridFoldTests: XCTestCase {

    // MARK: - at the boundary

    /// Sum, mean, count, min and max over Float64, Float32 and Int64 values at 2^24 - 1, 2^24, 2^24 + 1
    /// and 2^24 + 2^20 groups, against a host reference. The values are small multiples of 0.5, so every
    /// sum is exact in any order and the comparison is equality.
    func testAggregatesAt2To24Groups() throws {
        try requireRealGPU()
        for G in [(1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 24) + (1 << 20)] {
            try checkBoundary(groups: G)
        }
    }

    private func checkBoundary(groups G: Int) throws {
        let extra = 1 << 18
        let n = G + extra
        // Every group gets one row; the extra rows land on groups spread over the whole range, and one
        // row in 97 is null, so some single-row groups have no valid value and come back null.
        var keys = [Int32](repeating: 0, count: n)
        var iv = [Int64?](repeating: nil, count: n), dv = [Double?](repeating: nil, count: n)
        var fv = [Float?](repeating: nil, count: n)
        var cnt = [Int64](repeating: 0, count: G), isum = [Int64](repeating: 0, count: G)
        var imin = [Int64](repeating: .max, count: G), imax = [Int64](repeating: .min, count: G)
        for i in 0..<n {
            let k = i < G ? i : (i &* 7919) % G
            keys[i] = Int32(k)
            guard i % 97 != 5 else { continue }
            let x = Int64(i % 2001) - 1000
            iv[i] = x; dv[i] = Double(x) * 0.5; fv[i] = Float(x) * 0.5
            cnt[k] += 1; isum[k] += x
            imin[k] = Swift.min(imin[k], x); imax[k] = Swift.max(imax[k], x)
        }
        let gb = try MetalArray<Int32>(keys).groupBy(keyCount: G)
        let d = try MetalArray<Double>(dv), f = try MetalArray<Float>(fv), l = try MetalArray<Int64>(iv)
        let empty = cnt.reduce(0) { $0 + ($1 == 0 ? 1 : 0) }

        // Group k must be null exactly when it has no valid value, and equal `want(k)` otherwise.
        func check<T: ArrowPrimitive & Equatable>(_ a: MetalArray<T>, _ what: String, _ want: (Int) -> T) {
            XCTAssertEqual(a.length, G, "\(what) length, G=\(G)")
            XCTAssertEqual(a.nullCount, empty, "\(what) null count, G=\(G)")
            withExtendedLifetime(a) {
                let p = a.valuePointer
                let bm = a.validity?.typed(UInt8.self)
                for k in 0..<G {
                    let valid = bm.map { Bitmap.isSet($0, k) } ?? true
                    if valid != (cnt[k] > 0) || (valid && p[k] != want(k)) {
                        XCTFail("\(what), G=\(G), group \(k): valid \(valid) value \(p[k]), want \(cnt[k] > 0 ? "\(want(k))" : "null")")
                        return
                    }
                }
            }
        }
        let sum = { (k: Int) in Double(isum[k]) * 0.5 }
        let mean = { (k: Int) in Double(isum[k]) * 0.5 / Double(cnt[k]) }
        check(try gb.sumDouble(d), "f64 sum", sum)
        check(try gb.meanDouble(d), "f64 mean", mean)
        check(try gb.sumFloatAsDouble(f), "f32 sum", sum)
        check(try gb.meanFloat(f), "f32 mean", mean)
        check(try gb.sum(l), "i64 sum") { isum[$0] }
        check(try gb.mean(l), "i64 mean") { Double(isum[$0]) / Double(cnt[$0]) }
        for (what, c) in [("f64 count", try gb.countValid(d)), ("f32 count", try gb.countValid(f)),
                          ("i64 count", try gb.countValid(l))] {
            XCTAssertEqual(c.nullCount, 0, "\(what), G=\(G)")
            XCTAssertEqual(c.toRawArray(), cnt, "\(what), G=\(G)")
        }
        let minD = { (k: Int) in Double(imin[k]) * 0.5 }, maxD = { (k: Int) in Double(imax[k]) * 0.5 }
        check(try gb.min64(d), "f64 min", minD)
        check(try gb.max64(d), "f64 max", maxD)
        check(try gb.min(f), "f32 min") { Float(imin[$0]) * 0.5 }
        check(try gb.max(f), "f32 max") { Float(imax[$0]) * 0.5 }
        check(try gb.min64(l), "i64 min") { imin[$0] }
        check(try gb.max64(l), "i64 max") { imax[$0] }
        // The segmented 64-bit min/max, which reads its group from the grid like the sum.
        let seg = try gb.segments()
        check(try gb.min64(d, segments: seg), "f64 segmented min", minD)
        check(try gb.max64(l, segments: seg), "i64 segmented max") { imax[$0] }
    }

    // MARK: - the folded grid at small sizes

    /// Runs every per-group kernel twice, once on the plain grid and once with the fold forced on (rows
    /// of 7 threadgroups, so the last row is partial and holds threadgroups past the last group), and
    /// asserts the answers are the same. 3000 groups over 120,000 rows take the atomic counting sort
    /// with the per-group run sort (`cs_fix_tg`, one group holds 600+ rows) and the wide variance.
    func testFoldedGridMatchesPlainGrid() throws {
        try requireRealGPU()
        let K = 3000, n = 120_000
        var keys = [Int32](repeating: 0, count: n)
        var dv = [Double?](repeating: nil, count: n), iv = [Int64?](repeating: nil, count: n)
        var fv = [Float?](repeating: nil, count: n)
        var g = SystemRandomNumberGenerator()
        for i in 0..<n {
            keys[i] = i < 600 ? 0 : Int32((i &* 7919) % K)
            guard i % 13 != 4 else { continue }
            let x = Int64.random(in: -500...500, using: &g)
            iv[i] = x; dv[i] = Double(x) * 0.25; fv[i] = Float(x) * 0.25
        }
        let d = try MetalArray<Double>(dv), f = try MetalArray<Float>(fv), l = try MetalArray<Int64>(iv)
        let keyArray = try MetalArray<Int32>(keys)

        struct Answers: Equatable {
            var ord: [Int32] = []
            var sumD: [Double?] = [], meanD: [Double?] = [], sumF: [Double?] = [], meanF: [Double?] = []
            var minD: [Double?] = [], maxL: [Int64?] = [], varD: [Double?] = [], prodL: [Int64?] = []
            var prodF: [Double?] = [], listOffsets: [Int32] = [], listValues: [Int64?] = []
            var segMin: [Int64?] = [], segMax: [Int64?] = []
        }
        func run() throws -> Answers {
            let gb = try keyArray.groupBy(keyCount: K)            // a fresh GroupBy: segments are rebuilt
            let seg = try gb.segments()
            var a = Answers()
            a.ord = seg.ord.toRawArray()
            a.sumD = try gb.sumDouble(d, segments: seg).toArray()
            a.meanD = try gb.meanDouble(d, segments: seg).toArray()
            a.sumF = try gb.sumFloatAsDouble(f, segments: seg).toArray()
            a.meanF = try gb.meanFloat(f, segments: seg).toArray()
            a.minD = try gb.min64(d, segments: seg).toArray()
            a.maxL = try gb.max64(l, segments: seg).toArray()
            a.varD = try gb.variance(d).toArray()
            a.prodL = try gb.productInt(l, segments: seg).toArray()
            a.prodF = try gb.productFloat(d, segments: seg).toArray()
            let list = try gb.list(l, segments: seg)
            a.listOffsets = withExtendedLifetime(list) {
                Array(UnsafeBufferPointer(start: list.offsets.typed(Int32.self), count: K + 1))
            }
            guard case .int64(let child) = list.values else { XCTFail("list child type"); return a }
            a.listValues = child.toArray()
            let mm = try gb.minMaxSegmented(l, segments: seg)
            a.segMin = mm.min.toArray(); a.segMax = mm.max.toArray()
            return a
        }

        let plain = try run()
        let (w, t) = (Dispatch.foldWidth, Dispatch.foldThreads)
        Dispatch.foldWidth = 7; Dispatch.foldThreads = 0
        defer { Dispatch.foldWidth = w; Dispatch.foldThreads = t }
        let grid = Dispatch.perGroupGrid(count: K)
        XCTAssertEqual(grid.width, 7)
        XCTAssertEqual(grid.height, (K + 6) / 7)
        let folded = try run()

        XCTAssertEqual(folded.ord, plain.ord, "group order")
        XCTAssertEqual(folded.sumD, plain.sumD, "f64 sum")
        XCTAssertEqual(folded.meanD, plain.meanD, "f64 mean")
        XCTAssertEqual(folded.sumF, plain.sumF, "f32 sum")
        XCTAssertEqual(folded.meanF, plain.meanF, "f32 mean")
        XCTAssertEqual(folded.minD, plain.minD, "f64 min")
        XCTAssertEqual(folded.maxL, plain.maxL, "i64 max")
        XCTAssertEqual(folded.varD, plain.varD, "f64 variance")
        XCTAssertEqual(folded.prodL, plain.prodL, "i64 product")
        XCTAssertEqual(folded.prodF, plain.prodF, "f64 product")
        XCTAssertEqual(folded.listOffsets, plain.listOffsets, "list offsets")
        XCTAssertEqual(folded.listValues, plain.listValues, "list values")
        XCTAssertEqual(folded.segMin, plain.segMin, "segmented min")
        XCTAssertEqual(folded.segMax, plain.segMax, "segmented max")

        // And the plain answers are right: the sums against the host.
        var sums = [Double](repeating: 0, count: K), seen = [Bool](repeating: false, count: K)
        for i in 0..<n { if let v = dv[i] { sums[Int(keys[i])] += v; seen[Int(keys[i])] = true } }
        XCTAssertEqual(plain.sumD, (0..<K).map { seen[$0] ? sums[$0] : nil }, "f64 sum vs host")
        XCTAssertEqual(plain.ord.count, n)
    }

    /// Below 2^32 threads the grid is the plain one-row grid, whatever the group count.
    func testPlainGridBelowTheLimit() {
        XCTAssertEqual(Dispatch.perGroupGrid(count: (1 << 24) - 1).height, 1)
        XCTAssertEqual(Dispatch.perGroupGrid(count: (1 << 24) - 1).width, (1 << 24) - 1)
        XCTAssertEqual(Dispatch.perGroupGrid(count: 1 << 24).height, 1 << 8)
        XCTAssertEqual(Dispatch.perGroupGrid(count: (1 << 24) + 1).height, (1 << 8) + 1)
        XCTAssertEqual(Dispatch.perGroupGrid(count: 0).width, 1)
        XCTAssertEqual(Dispatch.perGroupGrid(count: 1 << 24, threadsPerGroup: 32).height, 1)
    }
}
