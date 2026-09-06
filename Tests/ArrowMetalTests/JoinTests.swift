import XCTest
@testable import ArrowMetal

final class JoinTests: XCTestCase {
    // MARK: - Oracle

    /// One (left row, right row) pair packed into an Int64; a null right index becomes -1.
    private func pair(_ l: Int32, _ r: Int32?) -> Int64 { (Int64(l) << 32) | Int64(UInt32(bitPattern: r ?? -1)) }

    /// CPU reference: a dictionary of build rows per key. Null keys never match.
    private func oracle<K: Hashable>(_ left: [K?], _ right: [K?], _ kind: JoinKind) -> [Int64] {
        var byKey: [K: [Int32]] = [:]
        for (i, k) in right.enumerated() { if let k { byKey[k, default: []].append(Int32(i)) } }
        var out: [Int64] = []
        for (i, k) in left.enumerated() {
            let matches = k.flatMap { byKey[$0] } ?? []
            if matches.isEmpty {
                if kind == .left { out.append(pair(Int32(i), nil)) }
            } else {
                for m in matches { out.append(pair(Int32(i), m)) }
            }
        }
        return out.sorted()
    }

    /// The GPU result as the same sorted set of pairs (the join does not promise an order).
    private func pairs(_ res: (leftIndices: MetalArray<Int32>, rightIndices: MetalArray<Int32>)) -> [Int64] {
        XCTAssertEqual(res.leftIndices.length, res.rightIndices.length)
        let l = res.leftIndices.toRawArray(), r = res.rightIndices.toArray()
        return (0..<l.count).map { pair(l[$0], r[$0]) }.sorted()
    }

    // MARK: - Sizes, duplicates, nulls

    func testInnerAndLeftAcrossSizes() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let sizes = [0, 1, 1000, 300_003]
        for nL in sizes {
            for nR in sizes {
                // Half as many distinct keys as rows: duplicates on both sides, so many-to-many everywhere.
                let range = Int32(Swift.max(1, Swift.max(nL, nR) / 2))
                func keys(_ n: Int) -> [Int32?] {
                    (0..<n).map { _ in Int.random(in: 0..<16, using: &g) == 0 ? nil : Int32.random(in: 0..<range, using: &g) }
                }
                let lk = keys(nL), rk = keys(nR)
                let l = try MetalArray<Int32>(lk), r = try MetalArray<Int32>(rk)
                for kind in JoinKind.allCases {
                    let got = pairs(try hashJoin(left: l, right: r, kind: kind))
                    XCTAssertEqual(got, oracle(lk, rk, kind), "\(kind) join \(nL) x \(nR)")
                }
            }
        }
    }

    func testInt64Keys() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for (nL, nR) in [(1, 1), (1000, 17), (300_003, 1000), (1000, 300_003)] {
            // Keys far outside the 32-bit range, and pairs that share their low 32 bits.
            func keys(_ n: Int) -> [Int64?] {
                (0..<n).map { _ in
                    Int.random(in: 0..<16, using: &g) == 0 ? nil
                        : (Int64.random(in: 0..<500, using: &g) << 32) | Int64.random(in: 0..<500, using: &g)
                }
            }
            let lk = keys(nL), rk = keys(nR)
            let l = try MetalArray<Int64>(lk), r = try MetalArray<Int64>(rk)
            for kind in JoinKind.allCases {
                XCTAssertEqual(pairs(try hashJoin(left: l, right: r, kind: kind)), oracle(lk, rk, kind),
                               "int64 \(kind) join \(nL) x \(nR)")
            }
        }
    }

    func testDuplicateKeysManyToMany() throws {
        try requireRealGPU()
        let lk: [Int32?] = [1, 1, 2, 3, nil, 1]
        let rk: [Int32?] = [1, 2, 1, 1, nil, 2]
        let l = try MetalArray<Int32>(lk), r = try MetalArray<Int32>(rk)
        let inner = pairs(try hashJoin(left: l, right: r, kind: .inner))
        // Three left rows with key 1 x three right rows with key 1, plus 2 x two right rows with key 2.
        XCTAssertEqual(inner.count, 3 * 3 + 1 * 2)
        XCTAssertEqual(inner, oracle(lk, rk, .inner))
        let left = pairs(try hashJoin(left: l, right: r, kind: .left))
        XCTAssertEqual(left.count, inner.count + 2)          // key 3 and the null key come back unmatched
        XCTAssertEqual(left, oracle(lk, rk, .left))
        XCTAssertEqual(try hashJoin(left: l, right: r, kind: .left).rightIndices.nullCount, 2)
    }

    func testNullsNeverMatch() throws {
        try requireRealGPU()
        let lk: [Int32?] = [nil, nil, 7, nil]
        let rk: [Int32?] = [nil, nil, nil]
        let l = try MetalArray<Int32>(lk), r = try MetalArray<Int32>(rk)
        XCTAssertEqual(try hashJoin(left: l, right: r, kind: .inner).leftIndices.length, 0)
        let left = try hashJoin(left: l, right: r, kind: .left)
        XCTAssertEqual(left.leftIndices.toRawArray(), [0, 1, 2, 3])
        XCTAssertEqual(left.rightIndices.toArray(), [nil, nil, nil, nil])
        // A null key on one side does not match a null key on the other, even when the values are equal.
        let rk2: [Int32?] = [nil, 7]
        let r2 = try MetalArray<Int32>(rk2)
        XCTAssertEqual(pairs(try hashJoin(left: l, right: r2, kind: .inner)), oracle(lk, rk2, .inner))
    }

    func testNoMatchesAndAllMatches() throws {
        try requireRealGPU()
        let n = 5000
        let lk: [Int32?] = (0..<n).map { Int32($0) }
        let disjoint: [Int32?] = (0..<n).map { Int32($0 + n) }
        let same: [Int32?] = (0..<n).map { Int32($0) }
        let l = try MetalArray<Int32>(lk)
        // No matches at all.
        XCTAssertEqual(try hashJoin(left: l, right: try MetalArray<Int32>(disjoint), kind: .inner).leftIndices.length, 0)
        let leftJoin = try hashJoin(left: l, right: try MetalArray<Int32>(disjoint), kind: .left)
        XCTAssertEqual(leftJoin.leftIndices.length, n)
        XCTAssertEqual(leftJoin.rightIndices.nullCount, n)
        // Every left row matches exactly one right row.
        for kind in JoinKind.allCases {
            let res = try hashJoin(left: l, right: try MetalArray<Int32>(same), kind: kind)
            XCTAssertEqual(pairs(res), (0..<n).map { pair(Int32($0), Int32($0)) })
        }
        // An empty build side: nothing for an inner join, every row for a left join.
        let empty = try MetalArray<Int32>([Int32]())
        XCTAssertEqual(try hashJoin(left: l, right: empty, kind: .inner).leftIndices.length, 0)
        XCTAssertEqual(try hashJoin(left: l, right: empty, kind: .left).rightIndices.nullCount, n)
    }

    func testSlicedInputs() throws {
        try requireRealGPU()
        let lk: [Int32?] = (0..<2000).map { Int32($0 % 100) }
        let rk: [Int32?] = (0..<2000).map { Int32($0 % 37) }
        let l = try MetalArray<Int32>(lk).slice(offset: 64, length: 1000)
        let r = try MetalArray<Int32>(rk).slice(offset: 128, length: 500)
        let lSlice = Array(lk[64..<1064]), rSlice = Array(rk[128..<628])
        for kind in JoinKind.allCases {
            XCTAssertEqual(pairs(try hashJoin(left: l, right: r, kind: kind)), oracle(lSlice, rSlice, kind), "\(kind)")
        }
    }

    // MARK: - Record batches

    func testRecordBatchJoin() throws {
        try requireRealGPU()
        let left = try MetalRecordBatch(names: ["id", "name"], columns: [
            .int32(try MetalArray<Int32>([1, 2, 3, 4, nil])),
            .string(try MetalStringArray(["a", "b", "c", "d", "e"])),
        ])
        let right = try MetalRecordBatch(names: ["id", "tag"], columns: [
            .int32(try MetalArray<Int32>([2, 2, 3, 9])),
            .string(try MetalStringArray(["x", "y", "z", "w"])),
        ])

        func rows(_ b: MetalRecordBatch) -> [String] {
            let id = b["id"]!.asInt32!.toArray(), name = b["name"]!.asString!.toArray()
            let rid = b["id_right"]!.asInt32!.toArray(), tag = b["tag"]!.asString!.toArray()
            func s<T>(_ v: T?) -> String { v.map { "\($0)" } ?? "null" }
            return (0..<b.length).map { "\(s(id[$0]))/\(s(name[$0]))/\(s(rid[$0]))/\(s(tag[$0]))" }.sorted()
        }

        let inner = try left.join(right, on: "id", rightKey: "id", kind: .inner)
        XCTAssertEqual(inner.names, ["id", "name", "id_right", "tag"])
        XCTAssertEqual(inner.length, 3)
        XCTAssertEqual(rows(inner), ["2/b/2/x", "2/b/2/y", "3/c/3/z"])

        let leftJoin = try left.join(right, on: "id", rightKey: "id", kind: .left)
        XCTAssertEqual(leftJoin.length, 6)
        XCTAssertEqual(rows(leftJoin), ["1/a/null/null", "2/b/2/x", "2/b/2/y", "3/c/3/z", "4/d/null/null", "null/e/null/null"])
        XCTAssertEqual(leftJoin["tag"]!.nullCount, 3)

        // Distinct key names keep their own names; a missing or mistyped key is an error.
        let renamed = try MetalRecordBatch(names: ["rid", "tag"], columns: right.columns)
        let j = try left.join(renamed, on: "id", rightKey: "rid", kind: .inner)
        XCTAssertEqual(j.names, ["id", "name", "rid", "tag"])
        XCTAssertThrowsError(try left.join(right, on: "nope", rightKey: "id", kind: .inner))
        XCTAssertThrowsError(try left.join(right, on: "name", rightKey: "tag", kind: .inner))
    }

    func testRecordBatchJoinLargeAgainstOracle() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        let nL = 100_000, nR = 30_000
        let lk = (0..<nL).map { _ in Int64.random(in: 0..<20_000, using: &g) }
        let rk = (0..<nR).map { _ in Int64.random(in: 0..<20_000, using: &g) }
        let left = try MetalRecordBatch(names: ["k", "v"], columns: [
            .int64(try MetalArray<Int64>(lk)), .int64(try MetalArray<Int64>((0..<nL).map { Int64($0) })),
        ])
        let right = try MetalRecordBatch(names: ["k", "w"], columns: [
            .int64(try MetalArray<Int64>(rk)), .int64(try MetalArray<Int64>((0..<nR).map { Int64(1000 + $0) })),
        ])
        let j = try left.join(right, on: "k", rightKey: "k", kind: .inner)
        XCTAssertEqual(j.names, ["k", "v", "k_right", "w"])
        let expected = oracle(lk.map { Optional($0) }, rk.map { Optional($0) }, .inner)
        XCTAssertEqual(j.length, expected.count)
        // Every result row must be a real pair: v is the left row index, w - 1000 the right row index.
        let v = j["v"]!.asInt64!.toRawArray(), w = j["w"]!.asInt64!.toRawArray()
        let k = j["k"]!.asInt64!.toRawArray(), kr = j["k_right"]!.asInt64!.toRawArray()
        var got = [Int64]()
        for i in 0..<j.length {
            XCTAssertEqual(k[i], kr[i])
            got.append(pair(Int32(v[i]), Int32(w[i] - 1000)))
        }
        XCTAssertEqual(got.sorted(), expected)
    }

    // MARK: - Throughput

    /// Inner join of many left rows against a build side with ~5 rows per key. Small by default;
    /// set ARROWMETAL_JOIN_BENCH=1 for the full 10M x 1M measurement.
    func testInnerJoinThroughput() throws {
        try requireRealGPU()
        let big = ProcessInfo.processInfo.environment["ARROWMETAL_JOIN_BENCH"] != nil
        let nL = big ? 10_000_000 : 200_000
        let nR = big ? 1_000_000 : 20_000
        let distinct = Int32(nR / 5)                       // 5 build rows per key -> 5 matches per left row
        var g = SystemRandomNumberGenerator()
        let lk = (0..<nL).map { _ in Int32.random(in: 0..<distinct, using: &g) }
        let rk = (0..<nR).map { Int32($0) % distinct }
        let l = try MetalArray<Int32>(lk), r = try MetalArray<Int32>(rk)
        _ = try hashJoin(left: l, right: r, kind: .inner)   // warm up shader compilation and the buffer pool
        let t0 = DispatchTime.now().uptimeNanoseconds
        let res = try hashJoin(left: l, right: r, kind: .inner)
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        XCTAssertEqual(res.leftIndices.length, nL * 5)
        // Spot-check that the pairs really share a key (the exhaustive comparison is done above at small sizes).
        let li = res.leftIndices.toRawArray(), ri = res.rightIndices.toRawArray()
        for _ in 0..<1000 {
            let i = Int.random(in: 0..<li.count, using: &g)
            XCTAssertEqual(lk[Int(li[i])], rk[Int(ri[i])])
        }
        let rowsPerSecond = Double(nL + nR) / seconds, pairsPerSecond = Double(res.leftIndices.length) / seconds
        print(String(format: "join %d x %d -> %d pairs in %.1f ms: %.1f M input rows/s, %.1f M pairs/s",
                     nL, nR, res.leftIndices.length, seconds * 1000, rowsPerSecond / 1e6, pairsPerSecond / 1e6))
    }
}
