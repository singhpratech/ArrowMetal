import XCTest
@testable import ArrowMetal

/// Sort options: `nullPlacement` per key and `FloatOrder.total` (IEEE 754 totalOrder), against a CPU
/// reference that implements both orders from their definitions.
///
/// The data is built to hit every value the two float orders disagree on — NaN of both signs and several
/// payloads, -0.0 and +0.0, both infinities, subnormals — plus duplicates, at null fractions 0, 0.1 and
/// 1.0, and at sizes on both sides of the kernel's thresholds: 8191/8192/8193 (the radix block), 70,001,
/// and 2,000,000 (past the 2^18 analysis threshold, where the pass-skipping and the key-inverting
/// `sorted()` path run).
final class SortOptionsTests: XCTestCase {

    // MARK: - the CPU reference

    /// The totalOrder key: the bit pattern with the sign-flip transform (arrow-rs `total_cmp`).
    static func totalKey(_ bits: UInt64, width: Int) -> UInt64 {
        let sign: UInt64 = 1 << UInt64(width - 1)
        let mask: UInt64 = width == 64 ? .max : (1 << UInt64(width)) - 1
        return (bits & sign) != 0 ? (~bits & mask) : (bits | sign)
    }

    /// The reference permutation. `key` is nil for a NaN in IEEE order (its own block), otherwise an
    /// order-preserving key; nulls are nil values.
    static func reference(count n: Int, isNull: (Int) -> Bool, isNaN: (Int) -> Bool, key: (Int) -> UInt64,
                          descending: Bool, nullsFirst: Bool, ieee: Bool) -> [UInt32] {
        var valueRows: [(UInt64, UInt32)] = [], nanRows: [UInt32] = [], nullRows: [UInt32] = []
        valueRows.reserveCapacity(n)
        for i in 0..<n {
            if isNull(i) { nullRows.append(UInt32(i)) }
            else if ieee && isNaN(i) { nanRows.append(UInt32(i)) }
            else { let k = key(i); valueRows.append((descending ? ~k : k, UInt32(i))) }
        }
        valueRows.sort { $0.0 != $1.0 ? $0.0 < $1.0 : $0.1 < $1.1 }
        let values = valueRows.map(\.1)
        // IEEE order: NaN rows sit next to the nulls, in input order, in both directions.
        return nullsFirst ? nullRows + nanRows + values : values + nanRows + nullRows
    }

    static func referenceDouble(_ v: [Double?], descending: Bool, nullsFirst: Bool, order: FloatOrder) -> [UInt32] {
        reference(count: v.count, isNull: { v[$0] == nil }, isNaN: { v[$0]!.isNaN },
                  key: { i in
                      let x = v[i]!
                      if order == .total { return totalKey(x.bitPattern, width: 64) }
                      return totalKey((x == 0 ? 0.0 : x).bitPattern, width: 64)
                  },
                  descending: descending, nullsFirst: nullsFirst, ieee: order == .ieee)
    }

    static func referenceFloat(_ v: [Float?], descending: Bool, nullsFirst: Bool, order: FloatOrder) -> [UInt32] {
        reference(count: v.count, isNull: { v[$0] == nil }, isNaN: { v[$0]!.isNaN },
                  key: { i in
                      let x = v[i]!
                      if order == .total { return totalKey(UInt64(x.bitPattern), width: 32) }
                      return totalKey(UInt64((x == 0 ? Float(0) : x).bitPattern), width: 32)
                  },
                  descending: descending, nullsFirst: nullsFirst, ieee: order == .ieee)
    }

    static func referenceInt(_ v: [Int64?], descending: Bool, nullsFirst: Bool) -> [UInt32] {
        reference(count: v.count, isNull: { v[$0] == nil }, isNaN: { _ in false },
                  key: { UInt64(bitPattern: v[$0]!) ^ (1 << 63) },
                  descending: descending, nullsFirst: nullsFirst, ieee: false)
    }

    // MARK: - data

    static let specialDoubles: [Double] = [
        .nan, -.nan, Double(bitPattern: 0x7FF0_0000_0000_0001), Double(bitPattern: 0xFFF8_0000_0000_0042),
        Double(bitPattern: 0x7FF8_0000_0000_0007), Double(bitPattern: 0xFFF0_0000_0000_0003),
        0.0, -0.0, .infinity, -.infinity, .leastNonzeroMagnitude, -.leastNonzeroMagnitude,
        .leastNormalMagnitude / 4, -.leastNormalMagnitude / 8, .greatestFiniteMagnitude, -.greatestFiniteMagnitude,
    ]
    static let specialFloats: [Float] = [
        .nan, -.nan, Float(bitPattern: 0x7F80_0001), Float(bitPattern: 0xFFC0_0042), Float(bitPattern: 0x7FC0_0009),
        0.0, -0.0, .infinity, -.infinity, .leastNonzeroMagnitude, -.leastNormalMagnitude / 4,
    ]

    struct LCG: RandomNumberGenerator {
        var s: UInt64
        mutating func next() -> UInt64 { s = s &* 6364136223846793005 &+ 1442695040888963407; return s ^ (s >> 29) }
    }

    static func doubles(_ n: Int, nullFraction: Double, seed: UInt64) -> [Double?] {
        var g = LCG(s: seed)
        return (0..<n).map { _ in
            if nullFraction > 0 && Double.random(in: 0..<1, using: &g) < nullFraction { return nil }
            switch Int.random(in: 0..<4, using: &g) {
            case 0: return specialDoubles[Int.random(in: 0..<specialDoubles.count, using: &g)]
            case 1: return Double(Int.random(in: -20...20, using: &g)) / 4        // duplicates
            default: return Double.random(in: -1e6...1e6, using: &g)
            }
        }
    }

    static func floats(_ n: Int, nullFraction: Double, seed: UInt64) -> [Float?] {
        var g = LCG(s: seed)
        return (0..<n).map { _ in
            if nullFraction > 0 && Double.random(in: 0..<1, using: &g) < nullFraction { return nil }
            switch Int.random(in: 0..<4, using: &g) {
            case 0: return specialFloats[Int.random(in: 0..<specialFloats.count, using: &g)]
            case 1: return Float(Int.random(in: -20...20, using: &g)) / 4
            default: return Float.random(in: -1e4...1e4, using: &g)
            }
        }
    }

    static let optionGrid: [(Bool, NullPlacement)] = [(false, .atEnd), (false, .atStart), (true, .atEnd), (true, .atStart)]

    // MARK: - checks

    /// argsort, sorted() (bit for bit) and top-k against the reference, for one column.
    func checkDouble(_ v: [Double?], orders: [FloatOrder] = [.ieee, .total], ks: [Int] = [1, 7, 100, 1500],
                     file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<Double>(v)
        for order in orders {
            for (desc, place) in Self.optionGrid {
                let what = "Double n=\(v.count) desc=\(desc) \(place) \(order)"
                let want = Self.referenceDouble(v, descending: desc, nullsFirst: place == .atStart, order: order)
                let got = try a.argsort(descending: desc, nullPlacement: place, floatOrder: order).toRawArray()
                XCTAssertEqual(got, want, what, file: file, line: line)
                // The values: nulls where the reference puts them, every other row bit for bit (for
                // `.ieee` a zero or NaN row may be any of the tied rows, so compare by the reference's).
                let sorted = try a.sorted(descending: desc, nullPlacement: place, floatOrder: order).toArray()
                XCTAssertEqual(sorted.count, v.count, what, file: file, line: line)
                var bad = 0
                for (j, i) in want.enumerated() {
                    let w = v[Int(i)], s = sorted[j]
                    let same: Bool
                    switch (w, s) {
                    case (nil, nil): same = true
                    case let (x?, y?):
                        if order == .total { same = x.bitPattern == y.bitPattern }
                        else { same = x.isNaN ? y.isNaN : (x == y) }
                    default: same = false
                    }
                    if !same { bad += 1 }
                }
                XCTAssertEqual(bad, 0, "sorted() \(what)", file: file, line: line)
                for k in ks {
                    let top = try a.topK(k, largest: desc, nullPlacement: place, floatOrder: order).toRawArray()
                    XCTAssertEqual(top, Array(want.prefix(k)), "topK k=\(k) \(what)", file: file, line: line)
                }
            }
        }
    }

    func checkFloat(_ v: [Float?], ks: [Int] = [1, 100, 1500], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<Float>(v)
        for order in [FloatOrder.ieee, .total] {
            for (desc, place) in Self.optionGrid {
                let what = "Float n=\(v.count) desc=\(desc) \(place) \(order)"
                let want = Self.referenceFloat(v, descending: desc, nullsFirst: place == .atStart, order: order)
                XCTAssertEqual(try a.argsort(descending: desc, nullPlacement: place, floatOrder: order).toRawArray(),
                               want, what, file: file, line: line)
                if order == .total {
                    let sorted = try a.sorted(descending: desc, nullPlacement: place, floatOrder: order).toArray()
                    let wantBits = want.map { v[Int($0)]?.bitPattern }
                    XCTAssertEqual(sorted.map { $0?.bitPattern }, wantBits, "sorted() \(what)", file: file, line: line)
                }
                for k in ks {
                    XCTAssertEqual(try a.topK(k, largest: desc, nullPlacement: place, floatOrder: order).toRawArray(),
                                   Array(want.prefix(k)), "topK k=\(k) \(what)", file: file, line: line)
                }
            }
        }
    }

    // MARK: - tests

    func testTotalOrderAndNullPlacementAcrossSizes() throws {
        try requireRealGPU()
        var seed: UInt64 = 1
        for n in [0, 1, 2, 8191, 8192, 8193, 70_001] {
            for nf in [0.0, 0.1, 1.0] {
                seed += 1
                try checkDouble(Self.doubles(n, nullFraction: nf, seed: seed))
                try checkFloat(Self.floats(n, nullFraction: nf, seed: seed))
            }
        }
    }

    func testTwoMillionRows() throws {
        try requireRealGPU()
        // Past the analysis threshold: the pass-skipping readback, the partition's measured blocks and
        // the key-inverting `sorted()` all run. 0.05% nulls leaves ~1,000 null rows, so a top-k of 1,500
        // with the nulls first takes the nulls and then 500 rows from the radix select.
        try checkDouble(Self.doubles(2_000_000, nullFraction: 0.1, seed: 77), orders: [.total], ks: [100])
        try checkDouble(Self.doubles(2_000_000, nullFraction: 0.0005, seed: 78), orders: [.total], ks: [1500])
        try checkFloat(Self.floats(2_000_000, nullFraction: 0, seed: 79), ks: [100])
    }

    func testDefaultOptionsAreTheOldSort() throws {
        try requireRealGPU()
        for n in [8193, 300_001] {
            let v = Self.doubles(n, nullFraction: 0.1, seed: UInt64(n))
            let a = try MetalArray<Double>(v)
            for desc in [false, true] {
                XCTAssertEqual(try a.argsort(descending: desc).toRawArray(),
                               try a.argsort(descending: desc, nullPlacement: .atEnd, floatOrder: .ieee).toRawArray())
                XCTAssertEqual(try a.argsort(descending: desc).toRawArray(),
                               Self.referenceDouble(v, descending: desc, nullsFirst: false, order: .ieee))
                XCTAssertEqual(try a.topK(50, largest: desc).toRawArray(),
                               try a.topK(50, largest: desc, nullPlacement: .atEnd, floatOrder: .ieee).toRawArray())
            }
        }
    }

    func testSpecialValuesInTotalOrder() throws {
        try requireRealGPU()
        let v: [Double?] = [1, .nan, -0.0, nil, -.nan, 0.0, -.infinity, .infinity, -0.0, -1, nil]
        let a = try MetalArray<Double>(v)
        let asc = try a.sorted(floatOrder: .total).toArray().map { $0?.bitPattern }
        let want: [Double?] = [-.nan, -.infinity, -1, -0.0, -0.0, 0.0, 1, .infinity, .nan, nil, nil]
        XCTAssertEqual(asc, want.map { $0?.bitPattern })
        let descFirst = try a.sorted(descending: true, nullPlacement: .atStart, floatOrder: .total).toArray()
        let wantDesc: [Double?] = [nil, nil, .nan, .infinity, 1, 0.0, -0.0, -0.0, -1, -.infinity, -.nan]
        XCTAssertEqual(descFirst.map { $0?.bitPattern }, wantDesc.map { $0?.bitPattern })
        // IEEE order ties the zeros and keeps NaN next to the nulls.
        XCTAssertEqual(try a.argsort(descending: true).toRawArray(), [7, 0, 2, 5, 8, 9, 6, 1, 4, 3, 10])
        XCTAssertEqual(try a.argsort(descending: true, floatOrder: .total).toRawArray(), [1, 7, 0, 5, 2, 8, 9, 6, 4, 3, 10])
    }

    func testIntegerAndStringKeysTakeNullsFirst() throws {
        try requireRealGPU()
        var g = LCG(s: 5)
        for n in [1, 8193, 70_001] {
            let v: [Int64?] = (0..<n).map { _ in Int.random(in: 0..<10, using: &g) == 0 ? nil : Int64.random(in: -50...50, using: &g) }
            let a = try MetalArray<Int64>(v)
            for (desc, place) in Self.optionGrid {
                let want = Self.referenceInt(v, descending: desc, nullsFirst: place == .atStart)
                // `floatOrder` is ignored by an integer key.
                XCTAssertEqual(try a.argsort(descending: desc, nullPlacement: place, floatOrder: .total).toRawArray(), want)
                XCTAssertEqual(try a.topK(100, largest: desc, nullPlacement: place).toRawArray(), Array(want.prefix(100)))
            }
        }
        let s = try MetalStringArray(["b", nil, "a", nil, "c"])
        XCTAssertEqual(try AnyMetalArray.string(s).argsortIndices(descending: true, nullPlacement: .atStart).toRawArray(),
                       [1, 3, 4, 0, 2])
    }

    func testLexsortPerKeyOptions() throws {
        try requireRealGPU()
        var g = LCG(s: 9)
        for n in [8193, 70_001] {
            let k1: [Int64?] = (0..<n).map { _ in Int.random(in: 0..<8, using: &g) == 0 ? nil : Int64.random(in: 0..<4, using: &g) }
            let k2 = Self.doubles(n, nullFraction: 0.1, seed: UInt64(n))
            let c1 = AnyMetalArray.int64(try MetalArray<Int64>(k1))
            let c2 = AnyMetalArray.float64(try MetalArray<Double>(k2))
            for (d1, p1) in Self.optionGrid {
                for (d2, p2) in Self.optionGrid {
                    for o2 in [FloatOrder.ieee, .total] {
                        let got = try lexsortIndices([c1, c2], descending: [d1, d2], nullPlacements: [p1, p2],
                                                     floatOrders: [.ieee, o2]).toRawArray()
                        // Reference: rank of each row under key 2, then a stable sort by (key 1, rank 2).
                        let r2 = Self.referenceDouble(k2, descending: d2, nullsFirst: p2 == .atStart, order: o2)
                        var rank2 = [Int](repeating: 0, count: n)
                        // Rows tied under key 2 share a rank, so the tie falls through to row order.
                        var prev: UInt32? = nil, r = 0
                        for (j, i) in r2.enumerated() {
                            if let p = prev, !Self.tied(k2[Int(p)], k2[Int(i)], order: o2) { r = j }
                            if prev == nil { r = 0 }
                            rank2[Int(i)] = r; prev = i
                        }
                        let r1 = Self.referenceInt(k1, descending: d1, nullsFirst: p1 == .atStart)
                        var rank1 = [Int](repeating: 0, count: n)
                        prev = nil; r = 0
                        for (j, i) in r1.enumerated() {
                            if let p = prev, k1[Int(p)] != k1[Int(i)] { r = j }
                            rank1[Int(i)] = r; prev = i
                        }
                        let rows: [Int] = Array(0..<n)
                        let ordered: [Int] = rows.sorted { (a: Int, b: Int) -> Bool in
                            if rank1[a] != rank1[b] { return rank1[a] < rank1[b] }
                            if rank2[a] != rank2[b] { return rank2[a] < rank2[b] }
                            return a < b
                        }
                        let want: [UInt32] = ordered.map { UInt32($0) }
                        XCTAssertEqual(got, want, "lexsort n=\(n) \(d1) \(p1) / \(d2) \(p2) \(o2)")
                    }
                }
            }
        }
    }

    static func tied(_ a: Double?, _ b: Double?, order: FloatOrder) -> Bool {
        switch (a, b) {
        case (nil, nil): return true
        case let (x?, y?):
            if order == .total { return x.bitPattern == y.bitPattern }
            return (x.isNaN && y.isNaN) || x == y
        default: return false
        }
    }

    func testPlanJSONSortKeyOptions() throws {
        try requireRealGPU()
        let n = 70_001
        let v = Self.doubles(n, nullFraction: 0.05, seed: 3)
        let x = AnyMetalArray.float64(try MetalArray<Double>(v))
        let row = AnyMetalArray.uint32(try MetalArray<UInt32>((0..<UInt32(n)).map { $0 }))
        let batch = try MetalRecordBatch(names: ["x", "row"], columns: [x, row])
        let sources = ["t": PlanSource(name: "t", batch: batch)]
        let want = Self.referenceDouble(v, descending: true, nullsFirst: true, order: .total)
        let plans = [
            #"{"op":"sort","by":[["x",true,{"nulls":"first","float_order":"total"}]],"input":{"op":"scan","source":"t"}}"#,
            #"{"op":"sort","by":[{"column":"x","descending":true,"nulls":"first","float_order":"total"}],"input":{"op":"scan","source":"t"}}"#,
            #"{"op":"sort","nulls":"first","float_order":"total","by":[["x",true]],"input":{"op":"scan","source":"t"}}"#,
        ]
        for p in plans {
            let out = try PlanJSON.run(p, sources: sources)
            XCTAssertEqual(out["row"]!.asUInt32!.toRawArray(), want, p)
        }
        // sort + limit is the fused top-k in the physical plan.
        let topPlan = #"{"op":"limit","count":100,"input":{"op":"sort","by":[["x",true,{"nulls":"first","float_order":"total"}]],"input":{"op":"scan","source":"t"}}}"#
        XCTAssertTrue(try PlanJSON.explain(topPlan, sources: sources).contains("TOP"), "top-k fusion")
        XCTAssertEqual(try PlanJSON.run(topPlan, sources: sources)["row"]!.asUInt32!.toRawArray(), Array(want.prefix(100)))
        // A plan with no options parses and runs exactly as before.
        let old = #"{"op":"sort","by":[["x",true]],"input":{"op":"scan","source":"t"}}"#
        XCTAssertEqual(try PlanJSON.run(old, sources: sources)["row"]!.asUInt32!.toRawArray(),
                       Self.referenceDouble(v, descending: true, nullsFirst: false, order: .ieee))
        XCTAssertThrowsError(try PlanJSON.run(#"{"op":"sort","by":[["x",true,{"nulls":"middle"}]],"input":{"op":"scan","source":"t"}}"#, sources: sources))
        XCTAssertThrowsError(try PlanJSON.run(#"{"op":"sort","float_order":"weird","by":[["x",true]],"input":{"op":"scan","source":"t"}}"#, sources: sources))
    }

    func testSortKeyDescriptionUnchangedForDefaults() {
        XCTAssertEqual(SortKey("x").description, "x")
        XCTAssertEqual(SortKey("x", descending: true).description, "x DESC")
        XCTAssertEqual(SortKey("x", descending: true, nullsFirst: true, floatOrder: .total).description,
                       "x DESC NULLS FIRST TOTAL_ORDER")
    }
}
