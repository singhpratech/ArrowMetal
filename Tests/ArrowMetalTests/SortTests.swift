import XCTest
@testable import ArrowMetal

final class SortTests: XCTestCase {
    func check<T: ArrowPrimitive>(_ vals: [T?], file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(vals)
        let idx = try a.argsort().toRawArray()
        XCTAssertEqual(idx.count, vals.count, file: file, line: line)
        XCTAssertEqual(Set(idx).count, idx.count, "permutation", file: file, line: line)
        // expected: stable sort of non-null by total order, then nulls in original order
        let nonNull = vals.enumerated().filter { $0.element != nil }
        let sortedNonNull = nonNull.sorted { x, y in
            let a = x.element!, b = y.element!
            if a.isTotallyLess(b) { return true }
            if b.isTotallyLess(a) { return false }
            return x.offset < y.offset
        }.map { Int32($0.offset) }
        let nulls = vals.enumerated().filter { $0.element == nil }.map { Int32($0.offset) }
        XCTAssertEqual(idx, sortedNonNull + nulls, "\(T.self) n=\(vals.count)", file: file, line: line)
        let desc = try a.argsort(descending: true).toRawArray()
        let sortedDesc = nonNull.sorted { x, y in
            let a = x.element!, b = y.element!
            if b.isTotallyLess(a) { return true }
            if a.isTotallyLess(b) { return false }
            return x.offset < y.offset
        }.map { Int32($0.offset) }
        XCTAssertEqual(desc, sortedDesc + nulls, "desc", file: file, line: line)
    }

    func testArgsortAllTypes() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 2, 255, 4096, 4097, 100_003] {
            try check((0..<n).map { _ in Int.random(in: 0..<10, using: &g) == 0 ? nil : Int32.random(in: -1000...1000, using: &g) })
            try check((0..<n).map { _ in Int64.random(in: Int64.min...Int64.max, using: &g) as Int64? })
            try check((0..<n).map { _ in UInt64.random(in: 0...UInt64.max, using: &g) as UInt64? })
            try check((0..<n).map { _ in Int8.random(in: -100...100, using: &g) as Int8? })
            try check((0..<n).map { _ in UInt16.random(in: 0...1000, using: &g) as UInt16? })
            try check((0..<n).map { i in i % 50 == 0 ? Float.nan : Float.random(in: -1...1, using: &g) })
            try check((0..<n).map { i in i % 7 == 0 ? nil : (i % 50 == 1 ? -0.0 : Double.random(in: -1e6...1e6, using: &g)) })
        }
        // many duplicates: stability matters
        try check((0..<50_000).map { Int32($0 % 3) as Int32? })
    }

    func testSortedTopKAndBatch() throws {
        try requireRealGPU()
        let a = try MetalArray<Int32>([5, nil, 3, 9, 1, 9])
        XCTAssertEqual(try a.sorted().toArray(), [1, 3, 5, 9, 9, nil])
        XCTAssertEqual(try a.sorted(descending: true).toArray(), [9, 9, 5, 3, 1, nil])
        XCTAssertEqual(try a.topK(2).toRawArray(), [3, 5])
        XCTAssertEqual(try a.topK(2, largest: false).toRawArray(), [4, 2])
        let b = try MetalRecordBatch(names: ["k", "s"], columns: [.int32(a), .string(try MetalStringArray(["e", "n", "c", "i", "a", "i2"]))])
        let s = try b.sorted(by: "k")
        XCTAssertEqual(s["s"]!.asString!.toArray(), ["a", "c", "e", "i", "i2", "n"])
        // 1M sort throughput sanity
        var g = SystemRandomNumberGenerator()
        let big = try MetalArray<Int64>((0..<1_000_000).map { _ in Int64.random(in: -1_000_000...1_000_000, using: &g) })
        let sorted = try big.sorted().toRawArray()
        XCTAssertTrue(zip(sorted, sorted.dropFirst()).allSatisfy { $0 <= $1 })
    }
}

extension ArrowPrimitive {
    /// Arrow sort order for floats: every NaN is one value placed after +inf, -0.0 ties with 0.0; plain < otherwise.
    func isTotallyLess(_ o: Self) -> Bool {
        if let a = self as? Double, let b = o as? Double {
            if a.isNaN || b.isNaN { return !a.isNaN && b.isNaN }
            return a < b
        }
        if let a = self as? Float, let b = o as? Float {
            if a.isNaN || b.isNaN { return !a.isNaN && b.isNaN }
            return a < b
        }
        return self < o
    }
}
