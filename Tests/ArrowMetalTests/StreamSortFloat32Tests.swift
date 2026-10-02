import XCTest
@testable import ArrowMetal

/// The streaming external sort's merge over a Float32 key holds the order the per-batch GPU sort gives:
/// under `total` every bit pattern is its own value, signaling NaNs included, so the merge must compare
/// the Float32 bits, not a Double widening that quiets a signaling NaN and reorders it among the NaNs.
final class StreamSortFloat32Tests: XCTestCase {

    private func run(_ batches: [[UInt32?]], _ key: ExternalSortOperator.Key, limit: Int? = nil) throws -> [Int64] {
        var bs: [MetalRecordBatch] = []
        var id: Int64 = 0
        for b in batches {
            let x = try MetalArray<Float>(b.map { $0.map { Float(bitPattern: $0) } })
            let ids = try MetalArray<Int64>((0..<b.count).map { i -> Int64? in id + Int64(i) })
            id += Int64(b.count)
            bs.append(try MetalRecordBatch(names: ["x", "id"], columns: [.float32(x), .int64(ids)]))
        }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("am-sort-f32-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let sink = CollectingSink()
        let op = try ExternalSortOperator(keys: [key], sink: sink, scratch: dir, limit: limit)
        op.mergeFanIn = 2
        _ = try StreamingExecutor(source: ChunkedTableSource(bs)).run(op)
        guard let t = try sink.table() else { return [] }
        return try XCTUnwrap(t["id"]?.asInt64).toArray().map { $0 ?? -1 }
    }

    /// totalOrder's key of a Float32 bit pattern (ascending).
    private static func totalKey(_ b: UInt32) -> UInt32 { b & 0x8000_0000 != 0 ? ~b : b | 0x8000_0000 }

    func testTotalOrderSignalingNaNAcrossRuns() throws {
        try requireRealGPU()
        // Each value in its own batch, so every row is its own run and the merge alone orders them.
        let values: [UInt32] = [
            0x7FC0_0000,    // +qNaN
            0x7F80_0001,    // +sNaN, payload 1: below +qNaN in totalOrder
            0x7FA0_0000,    // +sNaN, payload 0x200000
            0xFF80_0001,    // -sNaN
            0xFFC0_0000,    // -qNaN
            0x7F80_0000,    // +inf
            0x3F80_0000,    // 1
            0x8000_0000,    // -0
        ]
        let batches = values.map { [$0] as [UInt32?] }
        for desc in [false, true] {
            let ids = (0..<values.count).sorted { a, b in
                let ka = Self.totalKey(values[a]), kb = Self.totalKey(values[b])
                return desc ? ka > kb : ka < kb
            }.map { Int64($0) }
            let key = ExternalSortOperator.Key("x", descending: desc, floatOrder: .total)
            XCTAssertEqual(try run(batches, key), ids, "desc=\(desc)")
            XCTAssertEqual(try run(batches, key, limit: 3), Array(ids.prefix(3)), "limit desc=\(desc)")
        }
    }
}
