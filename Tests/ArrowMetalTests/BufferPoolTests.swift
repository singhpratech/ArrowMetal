import XCTest
import Metal
@testable import ArrowMetal

/// The pool's accounting and eviction over long runs of give/take, the pattern of a long process.
final class BufferPoolTests: XCTestCase {
    func buffers(_ lengths: [Int]) throws -> [MTLBuffer] {
        let device = MetalContext.shared.device
        return lengths.map { device.makeBuffer(length: $0, options: .storageModeShared)! }
    }

    /// 300,000 give/take cycles leave one entry per pooled buffer, and evicting afterwards stays cheap.
    /// Before, `take` left every entry in the eviction order and eviction used `removeFirst()` on it.
    func testManyCyclesThenEvictionStaysFast() throws {
        let page = 16_384
        let pool = BufferPool(limitBytes: 8 * page)
        let bs = try buffers([page, 2 * page])
        let start = Date()
        for _ in 0..<300_000 {
            pool.give(bs[0])
            XCTAssertNotNil(pool.take(length: page))
        }
        // Keep the pool full so that every give evicts (the same buffer object stands in for fresh ones).
        let fill = try buffers(Array(repeating: page, count: 8))
        for b in fill { pool.give(b) }
        for _ in 0..<20_000 { pool.give(bs[1]) }
        XCTAssertLessThanOrEqual(pool.pooledBytes, pool.limitBytes)
        XCTAssertLessThan(Date().timeIntervalSince(start), 5.0)
    }

    /// Bytes pooled equal the lengths still takeable, through evictions of mixed lengths.
    func testAccountingThroughMixedEvictions() throws {
        let page = 16_384
        let pool = BufferPool(limitBytes: 10 * page)
        let lengths = [page, 2 * page, 3 * page]
        var g = SystemRandomNumberGenerator()
        for _ in 0..<5_000 {
            let len = lengths.randomElement(using: &g)!
            if Bool.random(using: &g) { pool.give(try buffers([len])[0]) } else { _ = pool.take(length: len) }
            XCTAssertLessThanOrEqual(pool.pooledBytes, pool.limitBytes)
        }
        let pooled = pool.pooledBytes
        var drained = 0
        for len in lengths { while let b = pool.take(length: len) { XCTAssertEqual(b.length, len); drained += len } }
        XCTAssertEqual(drained, pooled, "every pooled byte is takeable")
        XCTAssertEqual(pool.pooledBytes, 0)
    }
}
