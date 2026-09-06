import XCTest
import CArrowABI
@testable import ArrowMetal

final class BatchTests: XCTestCase {
    func testChainMatchesUnbatched() throws {
        let ctx = MetalContext.shared
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 1000, 300_000] {
            var vals: [Int64?] = []
            for _ in 0..<n { vals.append(Int.random(in: 0..<8, using: &g) == 0 ? nil : Int64.random(in: -500...500, using: &g)) }
            let a = try MetalArray<Int64>(vals)
            let keys = try MetalArray<Int32>((0..<n).map { Int32($0 % 7) })
            // Unbatched reference
            let r1 = try a.filter(try a.compare(.gt, 0).and(try a.compare(.lt, 400)))
            let s1 = try r1.multiply(2).sum()
            let c1 = try r1.cast(to: Float.self).max()
            let gb1 = try keys.groupBy(keyCount: 7).sum(a).toArray()
            // Batched
            let (r2len, s2, c2, gb2, mid) = try ctx.batch { () -> (Int, SumResult?, Float?, [Int64?], Int) in
                let m = try a.compare(.gt, 0).and(try a.compare(.lt, 400))
                let r = try a.filter(m)
                XCTAssertTrue(r.pending || n == 0 || !ctx.isBatching)
                let doubled = try r.multiply(2)
                let s = try doubled.sum()                 // forces a sync point, batch continues after
                XCTAssertTrue(ctx.isBatching)
                let c = try r.cast(to: Float.self).max()
                let gb = try keys.groupBy(keyCount: 7).sum(a).toArray()
                return (r.length, s, c, gb, doubled.nullCount)
            }
            XCTAssertFalse(ctx.isBatching)
            XCTAssertEqual(r2len, r1.length, "n=\(n)")
            XCTAssertEqual(s2, s1); XCTAssertEqual(c2, c1); XCTAssertEqual(gb2, gb1)
            XCTAssertEqual(mid, try r1.multiply(2).nullCount)
        }
    }

    func testDeferredLengthAndNullCountResolveAfterBatch() throws {
        let a = try MetalArray<Int32>((0..<10_000).map { $0 % 5 == 0 ? nil : Int32($0) })
        let out = try MetalContext.shared.batch { () -> MetalArray<Int32> in
            let r = try a.filter(where: .gt, 100)
            XCTAssertTrue(r.pending)
            return r
        }
        XCTAssertFalse(out.pending)
        let expected = (0..<10_000).filter { $0 % 5 != 0 && $0 > 100 }
        XCTAssertEqual(out.length, expected.count)
        XCTAssertEqual(out.nullCount, 0)
        XCTAssertEqual(out.toRawArray(), expected.map { Int32($0) })
        // Reading inside the batch also works (it syncs).
        try MetalContext.shared.batch {
            let r = try a.filter(where: .lt, 50)
            XCTAssertEqual(r.length, 40)
            XCTAssertEqual(r[0], 1)
            XCTAssertTrue(MetalContext.shared.isBatching, "sync point reopens the batch")
        }
    }

    func testTakeErrorSurfacesAtBatchEnd() throws {
        let a = try MetalArray<Int32>([1, 2, 3])
        XCTAssertThrowsError(try MetalContext.shared.batch { _ = try a.take(try MetalArray<Int32>([0, 7])) })
        XCTAssertFalse(MetalContext.shared.isBatching)
        XCTAssertEqual(try MetalContext.shared.batch { try a.take(try MetalArray<Int32>([2, 0])) }.toRawArray(), [3, 1])
    }

    func testPoolParksBuffersWhileBatching() throws {
        let ctx = MetalContext.shared
        let a = try MetalArray<Int64>((0..<100_000).map { Int64($0) })
        let result: SumResult? = try ctx.batch {
            var last: MetalArray<Int64>? = nil
            for i in 1...20 {
                // Each temp is dropped immediately; without parking its buffer could be reused by the next kernel
                // while the pending command buffer still reads/writes it.
                let t = try a.multiply(Int64(i))
                last = try (last ?? t).add(t)
            }
            return try last!.sum()
        }
        // sum_{i=1..20} of i*sum(a) with the chain: last = t1 + t1, then + t2, ... = 2*t1 + t2 + ... + t20
        let base = (0..<100_000).map { Int64($0) }.reduce(0, +)
        let expected = base * (2 + (2...20).reduce(0, +))
        XCTAssertEqual(result, .int(expected))
        XCTAssertEqual(ctx.openBatches.value, 0)
    }

    func testExportInsideBatchMaterialises() throws {
        let a = try MetalArray<Float>([1, 2, 3, 4])
        var schema = ArrowSchema(); var arr = ArrowArray()
        try MetalContext.shared.batch {
            let r = try a.filter(where: .ge, 2.5)
            r.exportArrowSchema(into: &schema); r.exportArrowArray(into: &arr)
        }
        XCTAssertEqual(arr.length, 2)
        let back = try importArrowArray(schema: &schema, array: &arr)
        XCTAssertEqual(back.array.asFloat32?.toRawArray(), [3, 4])
        schema.release?(&schema)
    }
}
