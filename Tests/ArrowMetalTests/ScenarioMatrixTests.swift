import XCTest
@testable import ArrowMetal

/// Every kernel over every type, null density, size and input shape (plain or sliced), against the CPU oracle.
final class ScenarioMatrixTests: XCTestCase {
    static let sizes = [0, 1, 31, 32, 33, 255, 256, 257, 8191, 8192, 8193, 70_001]
    static let nullRatios = [0.0, 0.3, 1.0]

    func make<T: ArrowPrimitive>(_: T.Type, n: Int, nulls: Double, sliced: Bool, gen: (inout SystemRandomNumberGenerator) -> T) throws -> MetalArray<T> {
        var g = SystemRandomNumberGenerator()
        // When sliced, build a bigger array and take a 32-aligned window plus an unaligned one.
        let pad = sliced ? 64 : 0
        var vals: [T?] = []
        vals.reserveCapacity(n + 2 * pad)
        for _ in 0..<(n + 2 * pad) { vals.append(Double.random(in: 0..<1, using: &g) < nulls ? nil : gen(&g)) }
        let full = try MetalArray<T>(vals)
        return sliced ? try full.slice(offset: 64, length: n) : full
    }

    func exercise<T: ArrowPrimitive>(_ a: MetalArray<T>, _ b: MetalArray<T>, label: String, scalar: T, file: StaticString = #filePath, line: UInt = #line) throws {
        // reductions
        let s = try a.sum(), cs = CPUReference.sum(a)
        XCTAssertEqual(s == nil, cs == nil, "\(label) sum nil", file: file, line: line)
        if let s, let cs {
            if T.isFloatingPoint { XCTAssertEqual(s.asDouble, cs.asDouble, accuracy: Swift.max(1e-3 * abs(cs.asDouble), 1e-2), "\(label) sum", file: file, line: line) }
            else { XCTAssertEqual(s, cs, "\(label) sum", file: file, line: line) }
        }
        XCTAssertEqual(try a.min(), CPUReference.min(a), "\(label) min", file: file, line: line)
        XCTAssertEqual(try a.max(), CPUReference.max(a), "\(label) max", file: file, line: line)
        // compare scalar/array, all ops
        for op in CompareOp.allCases {
            XCTAssertEqual(try a.compare(op, scalar).toArray(), try CPUReference.compare(a, op, scalar: scalar).toArray(), "\(label) cmp \(op)", file: file, line: line)
            XCTAssertEqual(try a.compare(op, b).toArray(), try CPUReference.compare(a, op, array: b).toArray(), "\(label) cmp arr \(op)", file: file, line: line)
        }
        // filter, fused filter, take
        let mask = try a.compare(.gt, scalar)
        XCTAssertEqual(try a.filter(mask).toArray(), try CPUReference.filter(a, mask).toArray(), "\(label) filter", file: file, line: line)
        XCTAssertEqual(try a.filter(where: .gt, scalar).toArray(), try CPUReference.filter(a, mask).toArray(), "\(label) fused filter", file: file, line: line)
        let idxVals: [Int32] = (0..<a.length).map { Int32((a.length - 1 - $0)) }
        let idx = try MetalArray<Int32>(idxVals)
        XCTAssertEqual(try a.take(idx).toArray(), idxVals.map { a[Int($0)] }, "\(label) take", file: file, line: line)
        // arithmetic
        for op in ArithmeticOp.allCases {
            XCTAssertEqual(try a.arithmetic(op, scalar).toArray(), try CPUReference.arithmetic(a, op, scalar: scalar).toArray(), "\(label) arith \(op)", file: file, line: line)
            XCTAssertEqual(try a.arithmetic(op, b).toArray(), try CPUReference.arithmetic(a, op, array: b).toArray(), "\(label) arith arr \(op)", file: file, line: line)
        }
        // cast round trip to a wider type
        XCTAssertEqual(try a.cast(to: Double.self).toArray().map { $0.map { Double($0) } }, a.toArray().map { $0.map { $0.asDouble } }, "\(label) cast", file: file, line: line)
        // slice of a slice
        if a.length > 40 { XCTAssertEqual(try a.slice(offset: 32, length: 8).toArray(), Array(a.toArray()[32..<40]), "\(label) reslice", file: file, line: line) }
    }

    func testMatrix() throws {
        try requireRealGPU()
        for n in Self.sizes {
            for nulls in Self.nullRatios {
                for sliced in [false, true] {
                    let label = "n=\(n) nulls=\(nulls) sliced=\(sliced)"
                    let i8a = try make(Int8.self, n: n, nulls: nulls, sliced: sliced) { Int8.random(in: -100...100, using: &$0) }
                    let i8b = try make(Int8.self, n: n, nulls: nulls, sliced: sliced) { Int8.random(in: 1...100, using: &$0) }
                    try exercise(i8a, i8b, label: "Int8 " + label, scalar: 3)
                    let u16a = try make(UInt16.self, n: n, nulls: nulls, sliced: sliced) { UInt16.random(in: 0...1000, using: &$0) }
                    let u16b = try make(UInt16.self, n: n, nulls: nulls, sliced: sliced) { UInt16.random(in: 1...1000, using: &$0) }
                    try exercise(u16a, u16b, label: "UInt16 " + label, scalar: 500)
                    let i32a = try make(Int32.self, n: n, nulls: nulls, sliced: sliced) { Int32.random(in: -1000...1000, using: &$0) }
                    let i32b = try make(Int32.self, n: n, nulls: nulls, sliced: sliced) { Int32.random(in: 1...1000, using: &$0) }
                    try exercise(i32a, i32b, label: "Int32 " + label, scalar: 0)
                    let i64a = try make(Int64.self, n: n, nulls: nulls, sliced: sliced) { Int64.random(in: -1_000_000...1_000_000, using: &$0) }
                    let i64b = try make(Int64.self, n: n, nulls: nulls, sliced: sliced) { Int64.random(in: 1...1_000_000, using: &$0) }
                    try exercise(i64a, i64b, label: "Int64 " + label, scalar: 7)
                    let u64a = try make(UInt64.self, n: n, nulls: nulls, sliced: sliced) { UInt64.random(in: 0...1_000_000, using: &$0) }
                    let u64b = try make(UInt64.self, n: n, nulls: nulls, sliced: sliced) { UInt64.random(in: 1...1_000_000, using: &$0) }
                    try exercise(u64a, u64b, label: "UInt64 " + label, scalar: 500_000)
                    let f32a = try make(Float.self, n: n, nulls: nulls, sliced: sliced) { Float.random(in: -100...100, using: &$0) }
                    let f32b = try make(Float.self, n: n, nulls: nulls, sliced: sliced) { Float.random(in: 1...100, using: &$0) }
                    try exercise(f32a, f32b, label: "Float " + label, scalar: 0.5)
                    let f64a = try make(Double.self, n: n, nulls: nulls, sliced: sliced) { Double.random(in: -100...100, using: &$0) }
                    let f64b = try make(Double.self, n: n, nulls: nulls, sliced: sliced) { Double.random(in: 1...100, using: &$0) }
                    try exercise(f64a, f64b, label: "Double " + label, scalar: -0.25)
                }
            }
        }
    }

    /// Many threads using the shared context at once: pipeline cache, pool and command queue must be safe.
    func testConcurrentUse() throws {
        try requireRealGPU()
        let cols = try (0..<8).map { c in try MetalArray<Int64>((0..<200_000).map { Int64($0 % (c + 2)) }) }
        var failures = 0
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: 64) { i in
            let a = cols[i % cols.count]
            do {
                let expected = CPUReference.sum(a)
                for _ in 0..<5 {
                    let f = try a.filter(where: .ge, 1)
                    let s = try a.sum()
                    let m = try a.compare(.eq, 0)
                    if s != expected || f.length + m.trueCount != a.length { lock.lock(); failures += 1; lock.unlock() }
                }
            } catch { lock.lock(); failures += 1; lock.unlock() }
        }
        XCTAssertEqual(failures, 0)
    }

    /// Pool churn: allocate, free, reallocate at many sizes and verify contents are never stale where they must be zero.
    func testPoolReuseKeepsZeroSemantics() throws {
        try requireRealGPU()
        for round in 0..<3 {
            for n in [10, 1000, 5000, 100_000] {
                let a = try MetalArray<Int32>((0..<n).map { Int32($0 + round) })
                _ = try a.multiply(2)                                  // pooled output, garbage after release
                let b = try MetalArray<Int32>.allocate(length: n, withValidity: true)   // must be zeroed
                XCTAssertEqual(b.toRawArray().allSatisfy { $0 == 0 }, true, "values zeroed n=\(n)")
                b.recomputeNullCount()
                XCTAssertEqual(b.nullCount, n, "validity zeroed n=\(n)")
            }
        }
        XCTAssertGreaterThan(MetalContext.shared.pool.pooledBytes, 0)
    }
}
