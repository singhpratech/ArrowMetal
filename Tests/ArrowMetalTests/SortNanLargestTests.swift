import XCTest
@testable import ArrowMetal

/// `FloatOrder.nanLargest` (NaN one value above +inf in both directions, -0.0 tied with +0.0: the order
/// Polars sorts floats in), and the per-key sort options of window `order_by` keys and of the streaming
/// external sort, each against a CPU reference.
///
/// The reference is `SortOptionsTests.reference`: every value gets an order-preserving key and the rows
/// sort by it stably, with the nulls in one block at the chosen end. Under `nanLargest` a NaN's key is the
/// one past +inf's and turns around with the rest when descending.
final class SortNanLargestTests: XCTestCase {

    typealias R = SortOptionsTests

    // MARK: - the CPU reference

    static func referenceDouble(_ v: [Double?], descending: Bool, nullsFirst: Bool, order: FloatOrder) -> [Int32] {
        guard order == .nanLargest else {
            return R.referenceDouble(v, descending: descending, nullsFirst: nullsFirst, order: order)
        }
        return R.reference(count: v.count, isNull: { v[$0] == nil }, isNaN: { v[$0]!.isNaN },
                           key: { i in
                               let x = v[i]!
                               if x.isNaN { return R.totalKey(0x7FF0_0000_0000_0001, width: 64) }
                               return R.totalKey((x == 0 ? 0.0 : x).bitPattern, width: 64)
                           },
                           descending: descending, nullsFirst: nullsFirst, ieee: false)
    }

    static func referenceFloat(_ v: [Float?], descending: Bool, nullsFirst: Bool, order: FloatOrder) -> [Int32] {
        guard order == .nanLargest else {
            return R.referenceFloat(v, descending: descending, nullsFirst: nullsFirst, order: order)
        }
        return R.reference(count: v.count, isNull: { v[$0] == nil }, isNaN: { v[$0]!.isNaN },
                           key: { i in
                               let x = v[i]!
                               if x.isNaN { return R.totalKey(0x7F80_0001, width: 32) }
                               return R.totalKey(UInt64((x == 0 ? Float(0) : x).bitPattern), width: 32)
                           },
                           descending: descending, nullsFirst: nullsFirst, ieee: false)
    }

    /// Two rows tie under `order` (nulls tie with nulls).
    static func tied(_ a: Double?, _ b: Double?, order: FloatOrder) -> Bool {
        R.tied(a, b, order: order)
    }

    /// Each row's rank under the reference order of `v`: the first position of its tie group.
    static func ranks(_ v: [Double?], descending: Bool, nullsFirst: Bool, order: FloatOrder) -> [Int] {
        let perm = referenceDouble(v, descending: descending, nullsFirst: nullsFirst, order: order)
        var rank = [Int](repeating: 0, count: v.count)
        var prev: Int32? = nil, r = 0
        for (j, i) in perm.enumerated() {
            if let p = prev, !tied(v[Int(p)], v[Int(i)], order: order) { r = j }
            rank[Int(i)] = r; prev = i
        }
        return rank
    }

    static let orders: [FloatOrder] = [.ieee, .total, .nanLargest]

    // MARK: - the kernels

    func checkDouble(_ v: [Double?], ks: [Int], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<Double>(v)
        for (desc, place) in R.optionGrid {
            let what = "Double n=\(v.count) desc=\(desc) \(place) nanLargest"
            let want = Self.referenceDouble(v, descending: desc, nullsFirst: place == .atStart, order: .nanLargest)
            XCTAssertEqual(try a.argsort(descending: desc, nullPlacement: place, floatOrder: .nanLargest).toRawArray(),
                           want, what, file: file, line: line)
            // A zero or NaN row may come back as any of the rows tied with it; everything else bit for bit.
            let sorted = try a.sorted(descending: desc, nullPlacement: place, floatOrder: .nanLargest).toArray()
            var bad = 0
            for (j, i) in want.enumerated() {
                switch (v[Int(i)], sorted[j]) {
                case (nil, nil): break
                case let (x?, y?): if !(x.isNaN ? y.isNaN : x == y) { bad += 1 }
                default: bad += 1
                }
            }
            XCTAssertEqual(bad, 0, "sorted() \(what)", file: file, line: line)
            for k in ks {
                XCTAssertEqual(try a.topK(k, largest: desc, nullPlacement: place, floatOrder: .nanLargest).toRawArray(),
                               Array(want.prefix(k)), "topK k=\(k) \(what)", file: file, line: line)
            }
        }
    }

    func checkFloat(_ v: [Float?], ks: [Int], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<Float>(v)
        for (desc, place) in R.optionGrid {
            let what = "Float n=\(v.count) desc=\(desc) \(place) nanLargest"
            let want = Self.referenceFloat(v, descending: desc, nullsFirst: place == .atStart, order: .nanLargest)
            XCTAssertEqual(try a.argsort(descending: desc, nullPlacement: place, floatOrder: .nanLargest).toRawArray(),
                           want, what, file: file, line: line)
            for k in ks {
                XCTAssertEqual(try a.topK(k, largest: desc, nullPlacement: place, floatOrder: .nanLargest).toRawArray(),
                               Array(want.prefix(k)), "topK k=\(k) \(what)", file: file, line: line)
            }
        }
    }

    func testNanLargestAcrossSizes() throws {
        try requireRealGPU()
        var seed: UInt64 = 100
        for n in [0, 1, 2, 8191, 8192, 8193, 70_001] {
            for nf in [0.0, 0.1, 1.0] {
                seed += 1
                try checkDouble(R.doubles(n, nullFraction: nf, seed: seed), ks: [1, 7, 100, 1500])
                try checkFloat(R.floats(n, nullFraction: nf, seed: seed), ks: [1, 100, 1500])
            }
        }
    }

    func testNanLargestTwoMillionRows() throws {
        try requireRealGPU()
        // Past the 2^18 analysis threshold: pass skipping, the measured partition and the key-inverting
        // `sorted()` with its NaN run at the front of a descending sort.
        try checkDouble(R.doubles(2_000_000, nullFraction: 0.1, seed: 177), ks: [100])
        try checkDouble(R.doubles(2_000_000, nullFraction: 0.0005, seed: 178), ks: [1500])
        try checkFloat(R.floats(2_000_000, nullFraction: 0, seed: 179), ks: [100])
    }

    /// The order Polars 1.44 and 2.0 give the same column (`pl.Series(v).sort(...)`, maintain_order).
    func testPolarsOrderOfSpecialValues() throws {
        try requireRealGPU()
        let v: [Double?] = [1, .nan, nil, -0.0, 0.0, -1, .infinity, -.infinity, -.nan, nil, 2, -0.0]
        let a = try MetalArray<Double>(v)
        func rows(_ desc: Bool, _ nullsFirst: Bool) throws -> [Int32] {
            try a.argsort(descending: desc, nullPlacement: nullsFirst ? .atStart : .atEnd,
                          floatOrder: .nanLargest).toRawArray()
        }
        XCTAssertEqual(try rows(false, true), [2, 9, 7, 5, 3, 4, 11, 0, 10, 6, 1, 8])
        XCTAssertEqual(try rows(false, false), [7, 5, 3, 4, 11, 0, 10, 6, 1, 8, 2, 9])
        XCTAssertEqual(try rows(true, true), [2, 9, 1, 8, 6, 10, 0, 3, 4, 11, 5, 7])
        XCTAssertEqual(try rows(true, false), [1, 8, 6, 10, 0, 3, 4, 11, 5, 7, 2, 9])
        // top-k, both selection paths, with and without the nulls in front.
        XCTAssertEqual(try a.topK(4, largest: true, nullPlacement: .atEnd, floatOrder: .nanLargest).toRawArray(), [1, 8, 6, 10])
        XCTAssertEqual(try a.topK(4, largest: true, nullPlacement: .atStart, floatOrder: .nanLargest).toRawArray(), [2, 9, 1, 8])
        XCTAssertEqual(try a.topK(4, largest: false, nullPlacement: .atEnd, floatOrder: .nanLargest).toRawArray(), [7, 5, 3, 4])
    }

    func testLexsortWithNanLargestKeys() throws {
        try requireRealGPU()
        var g = R.LCG(s: 19)
        for n in [8193, 70_001] {
            let k1: [Int64?] = (0..<n).map { _ in Int.random(in: 0..<8, using: &g) == 0 ? nil : Int64.random(in: 0..<4, using: &g) }
            let k2 = R.doubles(n, nullFraction: 0.1, seed: UInt64(n) + 5)
            let c1 = AnyMetalArray.int64(try MetalArray<Int64>(k1))
            let c2 = AnyMetalArray.float64(try MetalArray<Double>(k2))
            for (d1, p1) in R.optionGrid {
                for (d2, p2) in R.optionGrid {
                    let got = try lexsortIndices([c1, c2], descending: [d1, d2], nullPlacements: [p1, p2],
                                                 floatOrders: [.ieee, .nanLargest]).toRawArray()
                    let rank2 = Self.ranks(k2, descending: d2, nullsFirst: p2 == .atStart, order: .nanLargest)
                    let r1 = R.referenceInt(k1, descending: d1, nullsFirst: p1 == .atStart)
                    var rank1 = [Int](repeating: 0, count: n)
                    var prev: Int32? = nil, r = 0
                    for (j, i) in r1.enumerated() {
                        if let p = prev, k1[Int(p)] != k1[Int(i)] { r = j }
                        rank1[Int(i)] = r; prev = i
                    }
                    let want = (0..<n).sorted { (a: Int, b: Int) -> Bool in
                        if rank1[a] != rank1[b] { return rank1[a] < rank1[b] }
                        if rank2[a] != rank2[b] { return rank2[a] < rank2[b] }
                        return a < b
                    }.map { Int32($0) }
                    XCTAssertEqual(got, want, "lexsort n=\(n) \(d1) \(p1) / \(d2) \(p2)")
                }
            }
        }
    }

    func testPlanJSONNanLargest() throws {
        try requireRealGPU()
        let n = 70_001
        let v = R.doubles(n, nullFraction: 0.05, seed: 13)
        let x = AnyMetalArray.float64(try MetalArray<Double>(v))
        let row = AnyMetalArray.int32(try MetalArray<Int32>((0..<Int32(n)).map { $0 }))
        let sources = ["t": PlanSource(name: "t", batch: try MetalRecordBatch(names: ["x", "row"], columns: [x, row]))]
        for (desc, nullsFirst) in [(true, false), (true, true), (false, true), (false, false)] {
            let want = Self.referenceDouble(v, descending: desc, nullsFirst: nullsFirst, order: .nanLargest)
            let nulls = nullsFirst ? "first" : "last"
            let sort = #"{"op":"sort","by":[["x",\#(desc),{"nulls":"\#(nulls)","float_order":"nan_largest"}]],"input":{"op":"scan","source":"t"}}"#
            XCTAssertEqual(try PlanJSON.run(sort, sources: sources)["row"]!.asInt32!.toRawArray(), want, sort)
            let top = #"{"op":"limit","count":100,"input":\#(sort)}"#
            XCTAssertTrue(try PlanJSON.explain(top, sources: sources).contains("TOP"), "top-k fusion")
            XCTAssertEqual(try PlanJSON.run(top, sources: sources)["row"]!.asInt32!.toRawArray(), Array(want.prefix(100)), top)
        }
        XCTAssertEqual(SortKey("x", descending: true, nullsFirst: true, floatOrder: .nanLargest).description,
                       "x DESC NULLS FIRST NAN_LARGEST")
        XCTAssertEqual(FloatOrder(name: "nan_largest"), .nanLargest)
        XCTAssertEqual(FloatOrder.nanLargest.rawValue, "nan_largest")
    }

    // MARK: - window order_by keys

    /// row_number and rank over partitions, ordered by a float key with every option, against the
    /// reference order: rows sort by (partition, the key's reference rank, row), and a row's rank is the
    /// first position of its tie group inside its partition.
    func testWindowOrderByKeyOptions() throws {
        try requireRealGPU()
        for n in [5_000, 70_001] {
            let v = R.doubles(n, nullFraction: 0.1, seed: UInt64(n) + 31)
            var g = R.LCG(s: UInt64(n))
            let part: [Int32] = (0..<n).map { _ in Int32.random(in: 0..<7, using: &g) }
            let x = AnyMetalArray.float64(try MetalArray<Double>(v))
            let p = AnyMetalArray.int32(try MetalArray<Int32>(part))
            let sources = ["t": PlanSource(name: "t", batch: try MetalRecordBatch(names: ["x", "p"], columns: [x, p]))]
            for order in Self.orders {
                for (desc, place) in R.optionGrid {
                    let nulls = place == .atStart ? "first" : "last"
                    let key = #"["x",\#(desc),{"nulls":"\#(nulls)","float_order":"\#(order.rawValue)"}]"#
                    let plan = #"{"op":"window","input":{"op":"scan","source":"t"},"specs":[{"name":"rn","fn":"row_number","partition_by":["p"],"order_by":[\#(key)]},{"name":"rk","fn":"rank","partition_by":["p"],"order_by":[\#(key)]}]}"#
                    let out = try PlanJSON.run(plan, sources: sources)
                    let rn = out["rn"]!.asInt32!.toRawArray(), rk = out["rk"]!.asInt32!.toRawArray()
                    let keyRank = Self.ranks(v, descending: desc, nullsFirst: place == .atStart, order: order)
                    let sorted = (0..<n).sorted { (a: Int, b: Int) -> Bool in
                        if part[a] != part[b] { return part[a] < part[b] }
                        if keyRank[a] != keyRank[b] { return keyRank[a] < keyRank[b] }
                        return a < b
                    }
                    var wantRn = [Int32](repeating: 0, count: n), wantRk = [Int32](repeating: 0, count: n)
                    var start = 0, tieStart = 0
                    for (j, i) in sorted.enumerated() {
                        if j == 0 || part[sorted[j - 1]] != part[i] { start = j; tieStart = j }
                        else if keyRank[sorted[j - 1]] != keyRank[i] { tieStart = j }
                        wantRn[i] = Int32(j - start + 1); wantRk[i] = Int32(tieStart - start + 1)
                    }
                    let what = "n=\(n) \(order) desc=\(desc) \(place)"
                    XCTAssertEqual(rn, wantRn, "row_number \(what)")
                    XCTAssertEqual(rk, wantRk, "rank \(what)")
                }
            }
        }
    }

    func testWindowOrderByDefaultsUnchanged() throws {
        try requireRealGPU()
        let v: [Double?] = [3, nil, 1, .nan, 1, nil, -0.0, 0.0]
        let x = AnyMetalArray.float64(try MetalArray<Double>(v))
        let sources = ["t": PlanSource(name: "t", batch: try MetalRecordBatch(names: ["x"], columns: [x]))]
        // Nulls last and NaN next to them in both directions, as before the options existed.
        let asc = #"{"op":"window","input":{"op":"scan","source":"t"},"specs":[{"name":"rk","fn":"rank","order_by":[["x",false]]}]}"#
        XCTAssertEqual(try PlanJSON.run(asc, sources: sources)["rk"]!.asInt32!.toRawArray(), [5, 7, 3, 6, 3, 7, 1, 1])
        let desc = #"{"op":"window","input":{"op":"scan","source":"t"},"specs":[{"name":"rk","fn":"rank","order_by":[["x",true]]}]}"#
        XCTAssertEqual(try PlanJSON.run(desc, sources: sources)["rk"]!.asInt32!.toRawArray(), [1, 7, 2, 6, 2, 7, 4, 4])
        // The spec-level options are the defaults of its keys.
        let spec = #"{"op":"window","input":{"op":"scan","source":"t"},"specs":[{"name":"rk","fn":"rank","nulls":"first","float_order":"nan_largest","order_by":[["x",true]]}]}"#
        XCTAssertEqual(try PlanJSON.run(spec, sources: sources)["rk"]!.asInt32!.toRawArray(), [4, 1, 5, 3, 5, 1, 7, 7])
    }

    // MARK: - the streaming external sort

    static func batches(_ x: [Double?], _ k: [Int64?], chunk: Int) throws -> [MetalRecordBatch] {
        var out: [MetalRecordBatch] = []
        var lo = 0
        while lo < x.count {
            let hi = Swift.min(x.count, lo + chunk)
            let cols: [AnyMetalArray] = [.float64(try MetalArray<Double>(Array(x[lo..<hi]))),
                                         .int64(try MetalArray<Int64>(Array(k[lo..<hi]))),
                                         .int64(try MetalArray<Int64>((lo..<hi).map { Int64($0) }))]
            out.append(try MetalRecordBatch(names: ["x", "k", "id"], columns: cols))
            lo = hi
        }
        return out
    }

    func runSort(_ bs: [MetalRecordBatch], _ keys: [ExternalSortOperator.Key], limit: Int? = nil,
                 name: String) throws -> [Int64] {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("am-sort-opts-\(name)-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let sink = CollectingSink()
        let op = try ExternalSortOperator(keys: keys, sink: sink, scratch: dir, limit: limit)
        op.mergeFanIn = 4
        _ = try StreamingExecutor(source: ChunkedTableSource(bs)).run(op)
        guard let t = try sink.table() else { return [] }
        return try XCTUnwrap(t["id"]?.asInt64).toArray().map { $0 ?? -1 }
    }

    func testExternalSortKeyOptions() throws {
        try requireRealGPU()
        let n = 12_000
        let x = R.doubles(n, nullFraction: 0.1, seed: 55)
        var g = R.LCG(s: 56)
        let k: [Int64?] = (0..<n).map { _ in Int.random(in: 0..<6, using: &g) == 0 ? nil : Int64.random(in: 0..<5, using: &g) }
        let bs = try Self.batches(x, k, chunk: 700)            // 18 runs, merged in passes of 4
        for order in Self.orders {
            for (desc, place) in R.optionGrid {
                let key = ExternalSortOperator.Key("x", descending: desc, nullsFirst: place == .atStart, floatOrder: order)
                let want = Self.referenceDouble(x, descending: desc, nullsFirst: place == .atStart, order: order).map { Int64($0) }
                let what = "\(order) desc=\(desc) \(place)"
                XCTAssertEqual(try runSort(bs, [key], name: "one"), want, "full \(what)")
                // With a limit: the resident top-n, threshold pruning included.
                for limit in [1, 37, 900] {
                    XCTAssertEqual(try runSort(bs, [key], limit: limit, name: "lim"), Array(want.prefix(limit)),
                                   "limit=\(limit) \(what)")
                }
            }
        }
        // Two keys: an int64 key with its nulls first, then the float key.
        for order in Self.orders {
            for (d2, p2) in R.optionGrid {
                let keys = [ExternalSortOperator.Key("k", descending: true, nullsFirst: true),
                            ExternalSortOperator.Key("x", descending: d2, nullsFirst: p2 == .atStart, floatOrder: order)]
                let r1 = R.referenceInt(k, descending: true, nullsFirst: true)
                var rank1 = [Int](repeating: 0, count: n)
                var prev: Int32? = nil, r = 0
                for (j, i) in r1.enumerated() {
                    if let p = prev, k[Int(p)] != k[Int(i)] { r = j }
                    rank1[Int(i)] = r; prev = i
                }
                let rank2 = Self.ranks(x, descending: d2, nullsFirst: p2 == .atStart, order: order)
                let want = (0..<n).sorted { (a: Int, b: Int) -> Bool in
                    if rank1[a] != rank1[b] { return rank1[a] < rank1[b] }
                    if rank2[a] != rank2[b] { return rank2[a] < rank2[b] }
                    return a < b
                }.map { Int64($0) }
                let what = "two keys \(order) desc=\(d2) \(p2)"
                XCTAssertEqual(try runSort(bs, keys, name: "two"), want, what)
                XCTAssertEqual(try runSort(bs, keys, limit: 50, name: "two-lim"), Array(want.prefix(50)), "limit \(what)")
            }
        }
    }

    /// The top-n threshold never drops a row that belongs in the answer: a NaN the order puts first, a
    /// null with the nulls first, and a NaN threshold.
    func testTopNPruningKeepsRowsAheadOfTheThreshold() throws {
        try requireRealGPU()
        // The first batch fills the resident rows with plain values; the NaN and null rows arrive late.
        var x: [Double?] = (0..<4_000).map { (i: Int) -> Double? in Double(i % 500) }
        for i in 0..<1_000 {
            let v: Double? = i % 3 == 0 ? Double.nan : (i % 3 == 1 ? nil : Double(i))
            x.append(v)
        }
        let k = [Int64?](repeating: 1, count: x.count)
        let bs = try Self.batches(x, k, chunk: 500)
        for order in Self.orders {
            for (desc, place) in R.optionGrid {
                let key = ExternalSortOperator.Key("x", descending: desc, nullsFirst: place == .atStart, floatOrder: order)
                let want = Self.referenceDouble(x, descending: desc, nullsFirst: place == .atStart, order: order).map { Int64($0) }
                for limit in [10, 400] {
                    XCTAssertEqual(try runSort(bs, [key], limit: limit, name: "prune"), Array(want.prefix(limit)),
                                   "limit=\(limit) \(order) desc=\(desc) \(place)")
                }
            }
        }
        // A NaN resident at the cut (ieee: NaN after every value) must not reject later values.
        let y: [Double?] = [Double](repeating: .nan, count: 600) + (0..<600).map { Double($0) }
        let bs2 = try Self.batches(y, [Int64?](repeating: 0, count: y.count), chunk: 300)
        let want = Self.referenceDouble(y, descending: false, nullsFirst: false, order: .ieee).map { Int64($0) }
        XCTAssertEqual(try runSort(bs2, [.init("x")], limit: 20, name: "nan-cut"), Array(want.prefix(20)))
    }
}
